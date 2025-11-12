from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from models import Base
from database import engine, test_connection
from routes import auth, classification, images
import logging
import os
import asyncio
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Global flag for readiness probe
models_ready = False

async def ping_db():
    """Test database connection"""
    return test_connection()

async def ping_db_quick():
    """Quick database ping for readiness check"""
    return test_connection()

async def preload_models():
    """Preload AI models synchronously for startup"""
    global models_ready
    try:
        from routes.classification import get_model
        logger.info("Preloading AI models...")
        
        # Load both models
        pro_model = get_model(is_pro=True)
        logger.info("✓ Pro model preloaded")
        
        clinical_model = get_model(is_pro=False)
        logger.info("✓ Clinical model preloaded")
        
        logger.info("All models ready for inference")
        models_ready = True
        
    except Exception as e:
        logger.error(f"Error preloading models: {e}")
        logger.info("App will continue with on-demand model loading")
        models_ready = False

app = FastAPI(title="DermAI Backend", version="1.0.0")

@app.on_event("startup")
async def startup():
    logger.info("Starting up application...")
    
    # Test database connection and create tables
    if await ping_db():
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/verified successfully")
    else:
        logger.error("Failed to connect to database on startup")
    
    # Preload models
    await preload_models()

import os
from dotenv import load_dotenv

load_dotenv()

# Get allowed origins from environment or use localhost for development
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,capacitor://localhost,http://localhost:5173,https://derm-ai-black.vercel.app").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(classification.router, prefix="/api", tags=["classification"])
app.include_router(images.router, prefix="/api", tags=["images"])

# Mount static files directory for serving uploaded images
uploads_dir = "./uploads"
os.makedirs(uploads_dir, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")

@app.get("/")
def read_root():
    return {"message": "Welcome to DermAI Backend API"}

@app.get("/ready")
async def ready():
    """Readiness check - returns 200 only after DB + models are loaded"""
    ok = await ping_db_quick()
    return {"ok": ok and models_ready}

@app.get("/health")
def health_check():
    try:
        db_status = test_connection()
        
        # Check if models exist (don't load them during health check)
        from routes.classification import PRO_MODEL_PATH, CLINICAL_MODEL_PATH
        import os
        
        pro_model_exists = os.path.exists(PRO_MODEL_PATH)
        clinical_model_exists = os.path.exists(CLINICAL_MODEL_PATH)
        
        models_status = "ready" if (pro_model_exists and clinical_model_exists) else "missing"
        models_preloaded = getattr(app.state, 'models_preloaded', False)
        
        # App is healthy if database works and models exist (even if still loading)
        overall_status = "healthy" if db_status and models_status == "ready" else "degraded"
        
        return {
            "status": overall_status,
            "database": "connected" if db_status else "disconnected",
            "models": models_status,
            "models_preloaded": models_preloaded,
            "timestamp": str(__import__('datetime').datetime.now())
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": str(__import__('datetime').datetime.now())
        }