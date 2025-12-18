#!/usr/bin/env python3
"""
Entry point for the SAR Ship Detection API Server
"""
import uvicorn
from app.config import settings

if __name__ == "__main__":
    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║         SAR Ship Detection API Server                     ║
    ║                                                           ║
    ║   API Docs: http://localhost:{settings.PORT}/docs              ║
    ║   Health:   http://localhost:{settings.PORT}/health            ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # CRITICAL FIX: Use workers=1 to avoid model reloading issues
    # Multiple workers cause each worker to load models into GPU memory
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,  # Never use reload in Docker
        workers=1      # MUST be 1 when loading large models
    )