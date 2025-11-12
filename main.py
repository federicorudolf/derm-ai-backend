import logging
import os
from datetime import datetime

from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from dotenv import load_dotenv

from models import Base
from database import engine, test_connection
from routes import auth, classification, images

logger = logging.getLogger(__name__)

# -------- Readiness flag --------
models_ready = False

# -------- DB checks (sync) --------
def ping_db() -> bool:
    """Test database connection"""
    return test_connection()

def ping_db_quick() -> bool:
    """Quick database ping for readiness check"""
    return test_connection()

# -------- Model preload --------
def preload_models() -> None:
    """Preload AI models synchronously for startup"""
    global models_ready
    try:
        from routes.classification import get_model
        logger.info("Preloading AI models...")

        get_model(is_pro=True)
        logger.info("✓ Pro model preloaded")

        get_model(is_pro=False)
        logger.info("✓ Clinical model preloaded")

        logger.info("All models ready for inference")
        models_ready = True
        app.state.models_preloaded = True  # <- used by /health
    except Exception as e:
        logger.exception("Error preloading models")
        models_ready = False
        app.state.models_preloaded = False

# -------- App --------
app = FastAPI(title="DermAI Backend", version="1.0.0")

@app.on_event("startup")
def startup():
    logger.info("Starting up application...")

    # DB & tables
    if ping_db():
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/verified successfully")
    else:
        logger.error("Failed to connect to database on startup")

    # Preload models
    preload_models()

# -------- CORS --------
load_dotenv()
allowed_origins = [o.strip() for o in os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,capacitor://localhost,http://localhost:5173,https://derm-ai-black.vercel.app"
).split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# -------- Routers --------
app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(classification.router, prefix="/api", tags=["classification"])
app.include_router(images.router, prefix="/api", tags=["images"])

# -------- Static uploads --------
uploads_dir = "./uploads"
os.makedirs(uploads_dir, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")

# -------- Basic routes --------
@app.get("/")
def read_root():
    return {"message": "Welcome to DermAI Backend API"}

@app.get("/version")
def version():
    return {"version": app.version, "timestamp": datetime.now().isoformat()}

@app.get("/ready")
def ready(response: Response):
    """Return 200 only if DB + models are ready; else 503 for kube readinessProbe"""
    ok = ping_db_quick() and models_ready
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"ok": ok}

@app.get("/health")
def health_check(response: Response):
    try:
        db_status = ping_db()

        # Check if model files exist (no loading here)
        from routes.classification import PRO_MODEL_PATH, CLINICAL_MODEL_PATH
        pro_model_exists = os.path.exists(PRO_MODEL_PATH)
        clinical_model_exists = os.path.exists(CLINICAL_MODEL_PATH)

        models_status = "ready" if (pro_model_exists and clinical_model_exists) else "missing"
        models_preloaded = bool(getattr(app.state, "models_preloaded", False))

        overall_ok = db_status and (models_status == "ready")
        response.status_code = status.HTTP_200_OK if overall_ok else status.HTTP_206_PARTIAL_CONTENT

        return {
            "status": "healthy" if overall_ok else "degraded",
            "database": "connected" if db_status else "disconnected",
            "models": models_status,
            "models_preloaded": models_preloaded,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.exception("Health check error")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
