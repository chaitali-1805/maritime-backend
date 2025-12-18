"""
GPU Test Script - Verify your setup
"""
import torch
import sys

print("=" * 60)
print("GPU Test Script")
print("=" * 60)

print(f"\nPython version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

# Check CUDA
print(f"\nCUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Total memory: {props.total_memory / 1e9:.2f} GB")
        print(f"  Multi-processor count: {props.multi_processor_count}")
        print(f"  CUDA capability: {props.major}.{props.minor}")
    
    # Test GPU computation
    print("\nTesting GPU computation...")
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print("✓ GPU computation test passed!")
    
    # Test FP16
    print("\nTesting FP16...")
    x_fp16 = x.half()
    y_fp16 = y.half()
    z_fp16 = torch.matmul(x_fp16, y_fp16)
    print("✓ FP16 computation test passed!")
    
else:
    print("\n⚠️  CUDA is not available!")
    print("This means PyTorch was installed without CUDA support.")
    print("\nTo fix this, run:")
    print("pip uninstall torch torchvision")
    print("pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121")

print("\n" + "=" * 60)
