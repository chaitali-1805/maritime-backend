"""
API Routes

All API routes are registered in main.py
This file just groups the route modules together.
"""
from app.routes import health, cgan, detection, upload

__all__ = ["health", "cgan", "detection", "upload"]
