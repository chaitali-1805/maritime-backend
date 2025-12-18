"""
YOLO Detection Service - GPU optimized and modular
"""
import time
import random
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image
import numpy as np
import torch

from app.config import settings
from app.services.model_registry import model_registry, ModelType


class YOLOService:
    def __init__(self):
        self._loaded_models: Dict[str, Any] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[YOLO] Device: {self.device}")
        
        if not settings.USE_MOCK_MODELS:
            self._preload_models()
    
    def _preload_models(self):
        """Preload all available models"""
        print("[YOLO] Preloading models...")
        for model_id, config in model_registry.YOLO_MODELS.items():
            weight_path = model_registry.get_yolo_weight_path(model_id)
            if weight_path and weight_path.exists():
                self._load_model(model_id, weight_path)
            else:
                print(f"[YOLO WARNING] Weights not found for {model_id}: {weight_path}")
    
    def _load_model(self, model_id: str, weight_path: Path):
        """Load a YOLO model"""
        try:
            from ultralytics import YOLO
            print(f"[YOLO] Loading {model_id} from {weight_path}...")
            model = YOLO(str(weight_path))
            model.to(self.device)
            self._loaded_models[model_id] = model
            print(f"[YOLO] Successfully loaded {model_id}")
        except Exception as e:
            print(f"[YOLO ERROR] Failed to load {model_id}: {e}")
    
    async def detect(
        self,
        image_path: str,
        model_id: str,
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Run detection on an image"""
        start_time = time.time()
        
        config = model_registry.get_yolo_config(model_id)
        if not config:
            raise ValueError(f"Unknown model: {model_id}")
        
        image_type = config.model_type.value
        
        if settings.USE_MOCK_MODELS or model_id not in self._loaded_models:
            detections = await self._mock_detect(image_path, confidence_threshold)
        else:
            detections = await self._real_detect(image_path, model_id, confidence_threshold)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "image_type": image_type,
            "detections": detections,
            "processing_time_ms": processing_time
        }
    
    async def _mock_detect(self, image_path: str, confidence_threshold: float) -> List[Dict[str, Any]]:
        """Mock detection - generates random bounding boxes"""
        image = Image.open(image_path)
        img_width, img_height = image.size
        
        num_detections = random.randint(2, 6)
        detections = []
        
        for _ in range(num_detections):
            x = random.uniform(0.1, 0.7)
            y = random.uniform(0.1, 0.7)
            width = random.uniform(0.05, 0.15)
            height = random.uniform(0.05, 0.15)
            confidence = random.uniform(confidence_threshold, 0.98)
            
            detections.append({
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "confidence": round(confidence, 3),
                "class_name": "ship",
                "class_id": 0
            })
        
        return detections
    
    async def _real_detect(self, image_path: str, model_id: str, confidence_threshold: float) -> List[Dict[str, Any]]:
        """Real YOLO detection"""
        model = self._loaded_models[model_id]
        
        # Run inference
        results = model(image_path, conf=confidence_threshold, device=self.device, verbose=False)
        
        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get normalized coordinates
                    x1, y1, x2, y2 = box.xyxyn[0].tolist()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = result.names.get(class_id, "ship")
                    
                    detections.append({
                        "x": x1,
                        "y": y1,
                        "width": x2 - x1,
                        "height": y2 - y1,
                        "confidence": round(confidence, 3),
                        "class_name": class_name,
                        "class_id": class_id
                    })
        
        return detections


# Singleton instance
yolo_service = YOLOService()
