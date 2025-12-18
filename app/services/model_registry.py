"""
Dynamic Model Registry - Easy to add new models
"""
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from app.config import settings

class ModelType(str, Enum):
    SAR = "sar"
    OPTICAL = "optical"

@dataclass
class YOLOModelConfig:
    id: str
    name: str
    version: str
    model_type: ModelType
    weight_file: str
    description: str
    enabled: bool = True

@dataclass
class CGANModelConfig:
    id: str
    name: str
    weight_file: str
    description: str
    caption: str  # Text prompt for the model
    direction: str  # "a2b" or "b2a"
    image_size: int = 512
    enabled: bool = True

class ModelRegistry:
    """
    Central registry for all ML models
    Add new models by calling register_yolo_model() or updating YOLO_MODELS dict
    """
    
    # =========================================
    # YOLO MODELS - ADD NEW MODELS HERE
    # =========================================
    YOLO_MODELS: Dict[str, YOLOModelConfig] = {
        
        # YOLOv8 Models
        "yolov8_sar": YOLOModelConfig(
            id="yolov8_sar",
            name="YOLOv8 SAR",
            version="v8",
            model_type=ModelType.SAR,
            weight_file="yolo/yolov8_sar.pt",
            description="YOLOv8 trained on SAR ship images"
        ),
        "yolov8_optical": YOLOModelConfig(
            id="yolov8_optical",
            name="YOLOv8 Optical",
            version="v8",
            model_type=ModelType.OPTICAL,
            weight_file="yolo/yolov8_optical.pt",
            description="YOLOv8 trained on optical ship images"
        ),
      
        
        # =========================================
        # ADD NEW YOLO MODELS BELOW THIS LINE
        # =========================================
        # Example:
        # "yolov9_sar": YOLOModelConfig(
        #     id="yolov9_sar",
        #     name="YOLOv9 SAR",
        #     version="v9",
        #     model_type=ModelType.SAR,
        #     weight_file="yolo/yolov9_sar.pt",
        #     description="YOLOv9 trained on SAR ship images"
        # ),
    }
    
    # =========================================
    # CGAN MODEL CONFIG - CycleGAN-Turbo
    # =========================================
    CGAN_MODEL: CGANModelConfig = CGANModelConfig(
        id="cyclegan_turbo_sar2opt",
        name="CycleGAN-Turbo SAR to Optical",
        weight_file="cgan/sar2optical.pkl",
        description="CycleGAN-Turbo for SAR to Optical conversion",
        caption="optical satellite image of ships at sea",  # UPDATE THIS
        direction="a2b",  # UPDATE THIS based on your training
        image_size=512
    )
    
    def __init__(self):
        self._loaded_models: Dict[str, Any] = {}
    
    def list_yolo_models(self) -> List[Dict]:
        """List all registered YOLO models"""
        return [
            {
                "id": model.id,
                "name": model.name,
                "version": model.version,
                "type": model.model_type.value,
                "description": model.description,
                "enabled": model.enabled,
                "weights_exist": self._check_weights_exist(model.weight_file)
            }
            for model in self.YOLO_MODELS.values()
        ]
    
    def get_yolo_config(self, model_id: str) -> Optional[YOLOModelConfig]:
        """Get configuration for a specific YOLO model"""
        return self.YOLO_MODELS.get(model_id)
    
    def get_yolo_weight_path(self, model_id: str) -> Optional[Path]:
        """Get the full path to YOLO model weights"""
        config = self.get_yolo_config(model_id)
        if config:
            return settings.WEIGHTS_DIR / config.weight_file
        return None
    
    def is_cgan_available(self) -> bool:
        """Check if CGAN weights are available"""
        return self._check_weights_exist(self.CGAN_MODEL.weight_file)
    
    def get_cgan_weight_path(self) -> Path:
        """Get the full path to CGAN weights"""
        return settings.WEIGHTS_DIR / self.CGAN_MODEL.weight_file
    
    def _check_weights_exist(self, weight_file: str) -> bool:
        """Check if weight file exists"""
        weight_path = settings.WEIGHTS_DIR / weight_file
        return weight_path.exists()
    
    def register_yolo_model(self, config: YOLOModelConfig):
        """Dynamically register a new YOLO model"""
        self.YOLO_MODELS[config.id] = config
    
    def get_models_by_version(self, version: str) -> List[YOLOModelConfig]:
        """Get all models of a specific version (e.g., 'v8')"""
        return [m for m in self.YOLO_MODELS.values() if m.version == version]
    
    def get_models_by_type(self, model_type: ModelType) -> List[YOLOModelConfig]:
        """Get all models of a specific type (SAR or Optical)"""
        return [m for m in self.YOLO_MODELS.values() if m.model_type == model_type]
    
    def get_cgan_config(self) -> CGANModelConfig:
        """Get CGAN model configuration"""
        return self.CGAN_MODEL
    
    def update_cgan_config(self, caption: str = None, direction: str = None):
        """Update CGAN configuration dynamically"""
        if caption:
            self.CGAN_MODEL.caption = caption
        if direction:
            self.CGAN_MODEL.direction = direction

# Global registry instance
model_registry = ModelRegistry()
