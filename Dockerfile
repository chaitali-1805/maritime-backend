FROM nvcr.io/nvidia/pytorch:25.01-py3

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and other libraries
# Note: Ubuntu 24.04 uses libgl1 instead of libgl1-mesa-glx
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# CRITICAL: NumPy must be <2.0 for compatibility with PyTorch container
RUN pip uninstall -y numpy && \
    pip install --no-cache-dir \
    "numpy>=1.23.0,<2.0" \
    "diffusers==0.32.1" \
    "huggingface-hub>=0.20.0" \
    "transformers==4.47.1" \
    "peft==0.14.0" \
    "accelerate==1.2.1" \
    "fastapi>=0.104.0" \
    "uvicorn[standard]>=0.24.0" \
    "python-multipart>=0.0.6" \
    "pillow>=10.0.0" \
    "opencv-python-headless>=4.8.0" \
    "pydantic>=2.0.0" \
    "pydantic-settings>=2.0.0" \
    "python-dotenv>=1.0.0" \
    "aiofiles>=23.0.0" \
    "requests>=2.31.0" \
    "ultralytics>=8.0.0" \
    "scipy>=1.11.0" \
    "torchmetrics>=1.2.0"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p weights/yolo weights/cgan uploads logs output

# IMPORTANT: Verify img2img-turbo exists
# The img2img-turbo repo should be at weights/cgan/img2img-turbo/
RUN if [ ! -d "weights/cgan/img2img-turbo" ]; then \
        echo "WARNING: img2img-turbo not found at weights/cgan/img2img-turbo/"; \
        echo "The CGAN service will fall back to mock mode."; \
        echo "To fix: Place img2img-turbo repository at weights/cgan/img2img-turbo/"; \
    else \
        echo "✓ Found img2img-turbo at weights/cgan/img2img-turbo/"; \
    fi

# Expose port
EXPOSE 8000

# Environment variables for DGX Spark
ENV HOST=0.0.0.0
ENV PORT=8000
ENV DEVICE=cuda
ENV DTYPE=float16
ENV PYTHONUNBUFFERED=1
# DGX Spark specific: Enable unified memory awareness
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Health check using curl (simpler and more reliable)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "run.py"]