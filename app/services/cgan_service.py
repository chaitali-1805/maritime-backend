"""
CGAN Service for SAR to Optical conversion using CycleGAN-Turbo (img2img-turbo)
GPU-optimized and production-ready - Using ORIGINAL img2img-turbo implementation
"""
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from app.config import settings


class CGANService:
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32  # CycleGAN-Turbo uses float32
        self.caption: Optional[str] = None
        self.direction: str = "a2b"
        self.image_size: int = 640  # Your images are 640x640
        
        print(f"[CGAN] Device: {self.device}, Dtype: {self.dtype}")
        
        if not settings.USE_MOCK_MODELS:
            self._load_model()
    
    def _load_model(self):
        """Load CycleGAN-Turbo model from original img2img-turbo repository"""
        weight_path = settings.WEIGHTS_DIR / "cgan" / "sar2optical.pkl"
        
        if not weight_path.exists():
            print(f"[CGAN WARNING] Weights not found at {weight_path}")
            print("[CGAN WARNING] Falling back to mock mode")
            return
        
        try:
            # Add img2img-turbo src directory to Python path
            img2img_turbo_src = settings.WEIGHTS_DIR / "cgan" / "img2img-turbo" / "src"
            
            if not img2img_turbo_src.exists():
                raise FileNotFoundError(
                    f"img2img-turbo source not found at {img2img_turbo_src}\n"
                    f"Make sure you have cloned img2img-turbo repo to: {settings.WEIGHTS_DIR / 'cgan' / 'img2img-turbo'}"
                )
            
            # Add to path if not already there
            img2img_turbo_src_str = str(img2img_turbo_src)
            if img2img_turbo_src_str not in sys.path:
                sys.path.insert(0, img2img_turbo_src_str)
                print(f"[CGAN] Added to Python path: {img2img_turbo_src_str}")
            
            # Import from original repository
            from cyclegan_turbo import CycleGAN_Turbo
            
            print(f"[CGAN] Loading model from {weight_path}...")
            
            # Load model - same as your batch script
            self.model = CycleGAN_Turbo(pretrained_path=str(weight_path))
            self.model.eval()
            
            # Move to device
            if self.device.type == "cuda":
                self.model = self.model.cuda()
            else:
                self.model = self.model.cpu()
            
            # Enable memory efficient attention if available
            try:
                if hasattr(self.model, 'unet') and hasattr(self.model.unet, 'enable_xformers_memory_efficient_attention'):
                    self.model.unet.enable_xformers_memory_efficient_attention()
                    print("[CGAN] xFormers enabled for memory efficiency")
            except Exception as e:
                print(f"[CGAN] xFormers not available: {e}")
            
            # Set default caption and direction (same as your batch script)
            self.caption = "optical color satellite image"
            self.direction = "a2b"
            
            print(f"[CGAN] ✓ Model loaded successfully!")
            print(f"[CGAN] Caption: {self.caption}, Direction: {self.direction}")
            
        except ImportError as e:
            print(f"[CGAN ERROR] Failed to import from img2img-turbo: {e}")
            print(f"[CGAN ERROR] Make sure img2img-turbo is cloned to: {settings.WEIGHTS_DIR / 'cgan' / 'img2img-turbo'}")
            print("[CGAN] Falling back to mock mode")
            self.model = None
        except Exception as e:
            print(f"[CGAN ERROR] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            print("[CGAN] Falling back to mock mode")
            self.model = None
    
    def _build_transform(self, image_size: int = 640):
        """
        Build image transformation pipeline
        Note: Your images are already 640x640, so no resizing needed
        Just convert to tensor and normalize
        """
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    async def convert(self, sar_image_path: str, file_id: str) -> Dict[str, Any]:
        """Convert SAR image to Optical image"""
        start_time = time.time()
        
        if settings.USE_MOCK_MODELS or self.model is None:
            result = await self._mock_convert(sar_image_path, file_id)
        else:
            result = await self._real_convert(sar_image_path, file_id)
        
        result["processing_time_ms"] = int((time.time() - start_time) * 1000)
        return result
    
    async def _real_convert(self, sar_image_path: str, file_id: str) -> Dict[str, Any]:
        """Real CycleGAN-Turbo conversion using original implementation"""
        try:
            # Load SAR image
            input_image = Image.open(sar_image_path).convert("RGB")
            original_size = input_image.size
            
            print(f"[CGAN] Processing image: {original_size}")
            
            # Build transform (no resizing, just normalize)
            T_val = self._build_transform(self.image_size)
            
            # If image is not 640x640, resize it
            if original_size != (self.image_size, self.image_size):
                input_image = input_image.resize(
                    (self.image_size, self.image_size), 
                    Image.LANCZOS
                )
                print(f"[CGAN] Resized from {original_size} to {self.image_size}x{self.image_size}")
            
            # Convert to tensor and normalize
            x_t = T_val(input_image).unsqueeze(0)
            
            # Move to device
            if self.device.type == "cuda":
                x_t = x_t.cuda()
            else:
                x_t = x_t.cpu()
            
            print(f"[CGAN] Input tensor shape: {x_t.shape}, device: {x_t.device}")
            
            # Run inference (same as batch script)
            with torch.no_grad():
                output = self.model(x_t, direction=self.direction, caption=self.caption)
            
            print(f"[CGAN] Output tensor shape: {output.shape}")
            
            # Convert output to PIL Image (same as batch script)
            output_img = (output[0].cpu().permute(1, 2, 0).numpy() + 1) / 2
            output_img = (output_img * 255).astype('uint8')
            output_pil = Image.fromarray(output_img)
            
            # Resize back to original dimensions if needed
            if original_size != (self.image_size, self.image_size):
                output_pil = output_pil.resize(original_size, Image.LANCZOS)
                print(f"[CGAN] Resized output back to {original_size}")
            
            # Save optical image
            optical_filename = f"optical_{file_id}.png"
            optical_path = settings.OUTPUT_DIR / optical_filename
            output_pil.save(optical_path, quality=95)
            
            print(f"[CGAN] ✓ Saved to {optical_path}")
            
            # Clean up GPU memory
            del x_t, output, output_img
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            return {
                "optical_filename": optical_filename,
                "optical_path": str(optical_path)
            }
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"[CGAN ERROR] GPU OUT OF MEMORY: {e}")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            # Fall back to mock
            return await self._mock_convert(sar_image_path, file_id)
            
        except Exception as e:
            print(f"[CGAN ERROR] Inference failed: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to mock
            return await self._mock_convert(sar_image_path, file_id)
    
    async def _mock_convert(self, sar_image_path: str, file_id: str) -> Dict[str, Any]:
        """Mock conversion - pseudo-colored SAR image"""
        sar_image = Image.open(sar_image_path).convert("RGB")
        sar_array = np.array(sar_image)
        
        # Simple color transformation
        optical_array = np.zeros_like(sar_array)
        optical_array[:, :, 0] = np.clip(sar_array[:, :, 0] * 0.8 + 30, 0, 255)
        optical_array[:, :, 1] = np.clip(sar_array[:, :, 1] * 0.9 + 50, 0, 255)
        optical_array[:, :, 2] = np.clip(sar_array[:, :, 2] * 1.1 + 40, 0, 255)
        
        optical_image = Image.fromarray(optical_array.astype(np.uint8))
        
        optical_filename = f"optical_{file_id}.png"
        optical_path = settings.OUTPUT_DIR / optical_filename
        optical_image.save(optical_path)
        
        return {
            "optical_filename": optical_filename,
            "optical_path": str(optical_path)
        }


# Singleton instance
cgan_service = CGANService()