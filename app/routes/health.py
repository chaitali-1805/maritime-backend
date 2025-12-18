"""
Health check endpoints
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any
import torch

from app.config import settings

router = APIRouter()

class HealthResponse(BaseModel):
    status: str
    environment: str
    cuda_available: bool
    mock_mode: bool
    version: str

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API is running and healthy"""
    return HealthResponse(
        status="healthy",
        environment=settings.ENVIRONMENT,
        cuda_available=torch.cuda.is_available(),
        mock_mode=settings.USE_MOCK_MODELS,
        version="1.0.0"
    )

@router.get("/models")
async def list_models():
    """List all available models"""
    from app.services.model_registry import model_registry
    return {
        "yolo_models": model_registry.list_yolo_models(),
        "cgan_available": model_registry.is_cgan_available()
    }
