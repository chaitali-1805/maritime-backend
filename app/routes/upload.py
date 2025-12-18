"""
Simple file upload endpoint for temporary image storage
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import uuid
import aiofiles
from pathlib import Path

from app.config import settings

router = APIRouter()

class UploadResponse(BaseModel):
    success: bool
    path: str
    filename: str

@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file for temporary processing
    Returns the path that can be used for detection
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix or ".png"
    filename = f"upload_{file_id}{file_extension}"
    
    # Save uploaded file
    file_path = settings.UPLOAD_DIR / filename
    
    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)
    
    return UploadResponse(
        success=True,
        path=f"/uploads/{filename}",
        filename=filename
    )