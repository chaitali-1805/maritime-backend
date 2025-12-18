"""
Image utility functions
"""
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import base64
import io

def load_image(path: str) -> Image.Image:
    """Load an image from path"""
    return Image.open(path).convert("RGB")

def save_image(image: Image.Image, path: str) -> str:
    """Save an image to path"""
    image.save(path)
    return path

def resize_image(
    image: Image.Image,
    max_size: Tuple[int, int] = (1024, 1024)
) -> Image.Image:
    """Resize image while maintaining aspect ratio"""
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image

def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode()

def base64_to_image(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))

def draw_bounding_boxes(
    image: Image.Image,
    boxes: list,
    color: str = "red",
    width: int = 2
) -> Image.Image:
    """Draw bounding boxes on an image"""
    from PIL import ImageDraw
    
    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size
    
    for box in boxes:
        x = box["x"] * img_width
        y = box["y"] * img_height
        w = box["width"] * img_width
        h = box["height"] * img_height
        
        draw.rectangle(
            [x, y, x + w, y + h],
            outline=color,
            width=width
        )
        
        # Draw confidence label
        conf = box.get("confidence", 0)
        label = f"{box.get('class_name', 'ship')} {conf:.2f}"
        draw.text((x, y - 15), label, fill=color)
    
    return image
