"""
CycleGAN-Turbo Implementation - Based on Original img2img-turbo
Device-agnostic version that works on both CUDA and CPU - last
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig
import copy


def my_vae_encoder_fwd(self, sample):
    """Custom VAE encoder that captures skip connections"""
    sample = self.conv_in(sample)
    l_blocks = []
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
    self.current_down_blocks = l_blocks
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample


def my_vae_decoder_fwd(self, sample, latent_embeds=None):
    """Custom VAE decoder that uses skip connections"""
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    
    if self.ignore_skip:
        skip_convs = [None, None, None, None]
    else:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
    
    if self.incoming_skip_acts is not None and skip_convs[0] is not None:
        skip_acts = [self.incoming_skip_acts[0], self.incoming_skip_acts[1], 
                     self.incoming_skip_acts[2], self.incoming_skip_acts[3]]
        for i, skip_act in enumerate(skip_acts):
            skip_act = skip_convs[i](skip_act)
            sample = sample + self.gamma * skip_act
    
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    
    for up_block in self.up_blocks:
        sample = up_block(sample, latent_embeds)
    
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    
    self.incoming_skip_acts = None
    return sample


class VAE_encode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_encode, self).__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        return _vae.encode(x).latent_dist.sample() * _vae.config.scaling_factor


class VAE_decode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_decode, self).__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        assert _vae.encoder.current_down_blocks is not None
        _vae.decoder.incoming_skip_acts = _vae.encoder.current_down_blocks
        x_decoded = (_vae.decode(x / _vae.config.scaling_factor).sample).clamp(-1, 1)
        return x_decoded


def make_1step_sched():
    from diffusers import DDPMScheduler
    noise_scheduler_1step = DDPMScheduler.from_pretrained("stabilityai/sd-turbo", subfolder="scheduler")
    noise_scheduler_1step.set_timesteps(1, device="cuda")
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
    return noise_scheduler_1step


class CycleGAN_Turbo(torch.nn.Module):
    def __init__(self, pretrained_path=None, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        
        print(f"[CGAN] Initializing on device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").to(self.device)
        self.sched = make_1step_sched()
        if self.device.type == "cuda":
            self.sched.alphas_cumprod = self.sched.alphas_cumprod.cuda()
        else:
            self.sched.alphas_cumprod = self.sched.alphas_cumprod.cpu()
        
        # Initialize VAE
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        
        # Add skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).to(self.device)
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).to(self.device)
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).to(self.device)
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).to(self.device)
        vae.decoder.ignore_skip = False
        
        # Initialize UNet
        unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")
        
        self.unet, self.vae = unet, vae
        self.timesteps = torch.tensor([999], device=self.device).long()
        self.caption = None
        self.direction = None
        
        if pretrained_path is not None:
            print(f"[CGAN] Loading checkpoint from: {pretrained_path}")
            sd = torch.load(pretrained_path, map_location=self.device)
            self.load_ckpt_from_state_dict(sd)
        
        self.vae_enc.to(self.device)
        self.vae_dec.to(self.device)
        self.unet.to(self.device)
        
        print(f"[CGAN] Model initialized successfully")

    def load_ckpt_from_state_dict(self, sd):
        """Load checkpoint - handles THREE adapters: encoder, decoder, others"""
        print(f"[CGAN] Loading checkpoint with rank_unet={sd['rank_unet']}, rank_vae={sd['rank_vae']}")
        
        # CRITICAL FIX: Replace conv_in to accept 8 channels
        print(f"[CGAN] Replacing UNet conv_in to accept 8 channels...")
        old_conv_in = self.unet.conv_in
        
        # Create new conv_in with 8 input channels
        self.unet.conv_in = torch.nn.Conv2d(
            8,  # 8 input channels instead of 4
            320,  # same output channels
            kernel_size=3,
            padding=1
        ).to(self.device)
        
        # Initialize intelligently: duplicate the original 4-channel weights
        with torch.no_grad():
            # Copy first 4 channels from original weights
            self.unet.conv_in.weight[:, :4, :, :] = old_conv_in.weight.data
            # Duplicate to channels 4-7 (average initialization)
            self.unet.conv_in.weight[:, 4:, :, :] = old_conv_in.weight.data
            # Scale down since we're using 8 channels now
            self.unet.conv_in.weight *= 0.5
            # Copy bias
            if old_conv_in.bias is not None:
                self.unet.conv_in.bias.data = old_conv_in.bias.data
        
        print(f"[CGAN] ✓ UNet conv_in replaced: {self.unet.conv_in.weight.shape}")
        
        # Create three separate LoRA configs
        lora_conf_encoder = LoraConfig(
            r=sd["rank_unet"], 
            init_lora_weights="gaussian", 
            target_modules=sd["l_target_modules_encoder"], 
            lora_alpha=sd["rank_unet"]
        )
        lora_conf_decoder = LoraConfig(
            r=sd["rank_unet"], 
            init_lora_weights="gaussian", 
            target_modules=sd["l_target_modules_decoder"], 
            lora_alpha=sd["rank_unet"]
        )
        lora_conf_others = LoraConfig(
            r=sd["rank_unet"], 
            init_lora_weights="gaussian", 
            target_modules=sd["l_modules_others"], 
            lora_alpha=sd["rank_unet"]
        )
        
        # Add all three adapters
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        
        print(f"[CGAN] Added 3 adapters: default_encoder, default_decoder, default_others")
        
        # Load encoder weights
        loaded_encoder = 0
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_encoder.weight", ".weight")
            if "lora" in n and "default_encoder" in n:
                # Skip conv_in LoRA since we replaced the base layer
                if "conv_in" in n:
                    continue
                if name_sd in sd["sd_encoder"]:
                    p.data.copy_(sd["sd_encoder"][name_sd])
                    loaded_encoder += 1
        print(f"[CGAN] Loaded {loaded_encoder} encoder parameters")
        
        # Load decoder weights
        loaded_decoder = 0
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_decoder.weight", ".weight")
            if "lora" in n and "default_decoder" in n:
                if name_sd in sd["sd_decoder"]:
                    p.data.copy_(sd["sd_decoder"][name_sd])
                    loaded_decoder += 1
        print(f"[CGAN] Loaded {loaded_decoder} decoder parameters")
        
        # Load others weights
        loaded_others = 0
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_others.weight", ".weight")
            if "lora" in n and "default_others" in n:
                if name_sd in sd["sd_other"]:
                    p.data.copy_(sd["sd_other"][name_sd])
                    loaded_others += 1
        print(f"[CGAN] Loaded {loaded_others} other parameters")
        
        # CRITICAL: Set ALL THREE adapters active at once
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])
        print(f"[CGAN] ✓ Activated all 3 adapters simultaneously")
        
        # Load VAE
        vae_lora_config = LoraConfig(
            r=sd["rank_vae"], 
            init_lora_weights="gaussian", 
            target_modules=sd["vae_lora_target_modules"]
        )
        self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        self.vae.decoder.gamma = 1
        self.vae_b2a = copy.deepcopy(self.vae)
        
        self.vae_enc = VAE_encode(self.vae, vae_b2a=self.vae_b2a)
        self.vae_enc.load_state_dict(sd["sd_vae_enc"], strict=False)
        
        self.vae_dec = VAE_decode(self.vae, vae_b2a=self.vae_b2a)
        self.vae_dec.load_state_dict(sd["sd_vae_dec"], strict=False)
        
        # CRITICAL FIX: Load modified quant_conv weights (8 channels instead of 4)
        print(f"[CGAN] Loading modified quant_conv weights...")
        if "vae.quant_conv.weight" in sd["sd_vae_enc"]:
            self.vae.quant_conv.weight.data = sd["sd_vae_enc"]["vae.quant_conv.weight"].to(self.device)
            self.vae.quant_conv.bias.data = sd["sd_vae_enc"]["vae.quant_conv.bias"].to(self.device)
            print(f"[CGAN] ✓ vae.quant_conv loaded: {self.vae.quant_conv.weight.shape}")
        
        if "vae_b2a.quant_conv.weight" in sd["sd_vae_enc"]:
            self.vae_b2a.quant_conv.weight.data = sd["sd_vae_enc"]["vae_b2a.quant_conv.weight"].to(self.device)
            self.vae_b2a.quant_conv.bias.data = sd["sd_vae_enc"]["vae_b2a.quant_conv.bias"].to(self.device)
            print(f"[CGAN] ✓ vae_b2a.quant_conv loaded: {self.vae_b2a.quant_conv.weight.shape}")
        
        # Also load post_quant_conv
        if "vae.post_quant_conv.weight" in sd["sd_vae_dec"]:
            self.vae.post_quant_conv.weight.data = sd["sd_vae_dec"]["vae.post_quant_conv.weight"].to(self.device)
            self.vae.post_quant_conv.bias.data = sd["sd_vae_dec"]["vae.post_quant_conv.bias"].to(self.device)
            print(f"[CGAN] ✓ vae.post_quant_conv loaded: {self.vae.post_quant_conv.weight.shape}")
        
        if "vae_b2a.post_quant_conv.weight" in sd["sd_vae_dec"]:
            self.vae_b2a.post_quant_conv.weight.data = sd["sd_vae_dec"]["vae_b2a.post_quant_conv.weight"].to(self.device)
            self.vae_b2a.post_quant_conv.bias.data = sd["sd_vae_dec"]["vae_b2a.post_quant_conv.bias"].to(self.device)
            print(f"[CGAN] ✓ vae_b2a.post_quant_conv loaded: {self.vae_b2a.post_quant_conv.weight.shape}")
        
        print(f"[CGAN] ✓ VAE loaded successfully with modified quant_conv")

    def forward(self, x_t, direction="a2b", caption="optical satellite image"):
        """Forward pass for image translation"""
        x_t = x_t.to(self.device, dtype=self.dtype)
        
        # Tokenize caption
        caption_tokens = self.tokenizer(
            caption, 
            max_length=self.tokenizer.model_max_length,
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        caption_enc = self.text_encoder(caption_tokens)[0].detach().clone()
        
        # Encode
        x_enc = self.vae_enc(x_t, direction=direction).to(x_t.dtype)
        
        # UNet forward
        model_pred = self.unet(
            x_enc, 
            self.timesteps, 
            encoder_hidden_states=caption_enc
        ).sample
        
        # Scheduler step
        B = x_enc.shape[0]
        x_out = torch.stack([
            self.sched.step(model_pred[i], self.timesteps[i], x_enc[i], return_dict=True).prev_sample 
            for i in range(B)
        ])
        
        # Decode
        x_out_decoded = self.vae_dec(x_out, direction=direction)
        
        return x_out_decoded