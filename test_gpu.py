#!/usr/bin/env python3
"""
Quick test script to verify CGAN model loading
"""
import torch
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, '/app')

from app.models.cyclegan_turbo_patched import CycleGAN_Turbo_Patched

def test_model_loading(checkpoint_path: str):
    """Test loading the CGAN model"""
    print("\n" + "="*80)
    print("TESTING CGAN MODEL LOADING")
    print("="*80 + "\n")
    
    if not Path(checkpoint_path).exists():
        print(f"❌ ERROR: Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        print(f"Loading model from: {checkpoint_path}")
        print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        print()
        
        # Try to load the model
        model = CycleGAN_Turbo_Patched(
            pretrained_path=checkpoint_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        print("\n✓ Model loaded successfully!")
        
        # Check adapters
        print("\nChecking adapters...")
        if hasattr(model.unet, 'peft_config'):
            adapters = list(model.unet.peft_config.keys())
            print(f"✓ Found {len(adapters)} adapters: {adapters}")
            
            # Try setting each adapter
            for adapter_name in adapters:
                try:
                    model.unet.set_adapter(adapter_name)
                    print(f"  ✓ Successfully set adapter: {adapter_name}")
                except Exception as e:
                    print(f"  ❌ Failed to set adapter {adapter_name}: {e}")
        else:
            print("⚠ No PEFT config found - adapters may not be loaded")
        
        print("\n" + "="*80)
        print("✓ MODEL LOADING TEST PASSED")
        print("="*80 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: Failed to load model")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n" + "="*80)
        print("❌ MODEL LOADING TEST FAILED")
        print("="*80 + "\n")
        
        return False


if __name__ == "__main__":
    checkpoint_path = "weights/cgan/sar2optical.pkl"
    
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    
    success = test_model_loading(checkpoint_path)
    sys.exit(0 if success else 1)