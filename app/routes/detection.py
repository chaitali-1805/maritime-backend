"""
YOLO detection endpoints
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import time

from app.config import settings
from app.services.yolo_service import YOLOService

router = APIRouter()
yolo_service = YOLOService()

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
    confidence: float
    class_name: str = "ship"

class DetectionRequest(BaseModel):
    image_path: str
    model_id: str
    confidence_threshold: Optional[float] = 0.5

class DetectionResponse(BaseModel):
    success: bool
    model_id: str
    image_type: str
    detections: List[BoundingBox]
    processing_time_ms: float
    annotated_image_url: Optional[str] = None

@router.post("/run", response_model=DetectionResponse)
async def run_detection(request: DetectionRequest):
    """
    Run YOLO detection on an image
    """
    try:
        # Convert URL path to filesystem path if needed
        image_path = request.image_path
        
        # If path starts with /uploads/ or /outputs/, convert to filesystem path
        if image_path.startswith("/uploads/"):
            filename = image_path.replace("/uploads/", "")
            image_path = str(settings.UPLOAD_DIR / filename)
        elif image_path.startswith("/outputs/"):
            filename = image_path.replace("/outputs/", "")
            image_path = str(settings.OUTPUT_DIR / filename)
        
        # Verify file exists
        if not Path(image_path).exists():
            raise HTTPException(
                status_code=404, 
                detail=f"{request.image_path} does not exist (resolved to: {image_path})"
            )
        
        result = await yolo_service.detect(
            image_path=image_path,
            model_id=request.model_id,
            confidence_threshold=request.confidence_threshold
        )
        
        return DetectionResponse(
            success=True,
            model_id=request.model_id,
            image_type=result["image_type"],
            detections=[BoundingBox(**det) for det in result["detections"]],
            processing_time_ms=result["processing_time_ms"],
            annotated_image_url=result.get("annotated_image_url")
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch")
async def run_batch_detection(
    sar_image_path: str,
    optical_image_path: str,
    model_id: str,
    confidence_threshold: float = 0.5
):
    """
    Run detection on both SAR and Optical images with corresponding models
    """
    # Determine which models to use
    base_model = model_id.replace("_sar", "").replace("_optical", "")
    sar_model_id = f"{base_model}_sar"
    optical_model_id = f"{base_model}_optical"
    
    # Convert paths if needed
    def resolve_path(path: str) -> str:
        if path.startswith("/uploads/"):
            return str(settings.UPLOAD_DIR / path.replace("/uploads/", ""))
        elif path.startswith("/outputs/"):
            return str(settings.OUTPUT_DIR / path.replace("/outputs/", ""))
        return path
    
    sar_path = resolve_path(sar_image_path)
    optical_path = resolve_path(optical_image_path)
    
    # Run both detections
    sar_result = await yolo_service.detect(
        image_path=sar_path,
        model_id=sar_model_id,
        confidence_threshold=confidence_threshold
    )
    
    optical_result = await yolo_service.detect(
        image_path=optical_path,
        model_id=optical_model_id,
        confidence_threshold=confidence_threshold
    )
    
    return {
        "success": True,
        "sar_detections": sar_result,
        "optical_detections": optical_result
    }