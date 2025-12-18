"""
Services - Core business logic

Import services directly:
    from app.services import CGANService, YOLOService, ModelRegistry
"""
from app.services.cgan_service import CGANService
from app.services.yolo_service import YOLOService
from app.services.model_registry import ModelRegistry

__all__ = ["CGANService", "YOLOService", "ModelRegistry"]
