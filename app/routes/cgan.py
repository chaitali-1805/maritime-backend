"""
CGAN (SAR to Optical) conversion endpoints
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import uuid
import aiofiles
from pathlib import Path

from app.config import settings
from app.services.cgan_service import CGANService

router = APIRouter()
cgan_service = CGANService()

class ConversionResponse(BaseModel):
    success: bool
    sar_image_url: str
    optical_image_url: str
    processing_time_ms: float

@router.post("/convert", response_model=ConversionResponse)
async def convert_sar_to_optical(file: UploadFile = File(...)):
    """
    Convert a SAR image to an Optical image using CGAN
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix or ".png"
    
    # Save uploaded file
    sar_filename = f"sar_{file_id}{file_extension}"
    sar_path = settings.UPLOAD_DIR / sar_filename
    
    async with aiofiles.open(sar_path, "wb") as f:
        content = await file.read()
        await f.write(content)
    
    # Process with CGAN
    try:
        result = await cgan_service.convert(str(sar_path), file_id)
        
        return ConversionResponse(
            success=True,
            sar_image_url=f"/uploads/{sar_filename}",
            optical_image_url=f"/outputs/{result['optical_filename']}",
            processing_time_ms=result["processing_time_ms"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
