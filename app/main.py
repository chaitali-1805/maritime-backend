"""
Main FastAPI application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routes import health, cgan, detection, upload

# Create FastAPI app
app = FastAPI(
    title="SAR Ship Detection API",
    description="API for SAR to Optical image conversion and ship detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving processed images
app.mount("/outputs", StaticFiles(directory=str(settings.OUTPUT_DIR)), name="outputs")
app.mount("/uploads", StaticFiles(directory=str(settings.UPLOAD_DIR)), name="uploads")

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(cgan.router, prefix="/api/cgan", tags=["CGAN"])
app.include_router(detection.router, prefix="/api/detect", tags=["Detection"])
app.include_router(upload.router, prefix="/api/upload", tags=["Upload"])

@app.on_event("startup")
async def startup_event():
    print("Starting SAR Ship Detection API...")
    print(f"Debug mode: {settings.DEBUG}")
    print(f"Using mock models: {settings.USE_MOCK_MODELS}")

@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down SAR Ship Detection API...")
