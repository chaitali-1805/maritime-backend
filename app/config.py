"""
Configuration settings for the ML backend
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    ENVIRONMENT: str = "development"
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    WEIGHTS_DIR: Path = BASE_DIR / "weights"
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    OUTPUT_DIR: Path = BASE_DIR / "outputs"
    
    # CORS - Add your frontend URLs here
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "https://*.vercel.app",
        "https://*.v0.dev",
        "*"  # Allow all for development - restrict in production
    ]
    
    # Model settings
    USE_MOCK_MODELS: bool = False  # Set to False when real models are ready
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.5
    
    class Config:
        env_file = ".env"
        extra = "allow"

    def setup_directories(self):
        """Create required directories if they don't exist"""
        self.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (self.WEIGHTS_DIR / "cgan").mkdir(exist_ok=True)
        (self.WEIGHTS_DIR / "yolo").mkdir(exist_ok=True)

settings = Settings()
settings.setup_directories()
