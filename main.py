from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from models import Base
from database import engine, test_connection
from routes import auth, classification, images
import logging
import os
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up application...")
    
    # Test database connection on startup
    if test_connection():
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/verified successfully")
    else:
        logger.error("Failed to connect to database on startup")
    
    # Preload models to avoid cold start delays
    try:
        from routes.classification import get_model
        logger.info("Preloading AI models...")
        
        # Load both models
        pro_model = get_model(is_pro=True)
        clinical_model = get_model(is_pro=False)
        
        logger.info("✓ Pro model preloaded")
        logger.info("✓ Clinical model preloaded")
        logger.info("All models ready for inference")
        
    except Exception as e:
        logger.error(f"Error preloading models: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")

app = FastAPI(title="DermAI Backend", version="1.0.0", lifespan=lifespan)

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

@app.get("/health")
def health_check():
    db_status = test_connection()
    return {
        "status": "healthy" if db_status else "unhealthy",
        "database": "connected" if db_status else "disconnected"
    }