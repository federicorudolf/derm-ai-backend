import os
import io
import uuid
import torch
import torch.nn as nn
import timm
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from datetime import date
from typing import Optional
import logging
import asyncio
import concurrent.futures
from functools import lru_cache
from auth import get_current_user
from models import User as UserModel, Picture, Diagnosis
from database import get_db

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

# Model configuration
PRO_MODEL_PATH = "./checkpoints/derm_densenet121_best.pth"
CLINICAL_MODEL_PATH = "./checkpoints/derm_densenet121_best_clinical.pth"
IMG_SIZE = 448
UPLOAD_DIR = "./uploads"

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Optimized device setup for deployment
# Force CPU for Railway deployment since GPUs aren't available
DEVICE = torch.device("cpu")
# Set number of threads for CPU inference optimization
torch.set_num_threads(2)  # Railway typically provides 2 CPU cores
print(f"Using device: {DEVICE} with {torch.get_num_threads()} threads")

# Optimized image preprocessing for faster CPU inference
# Reduced image size for faster processing on CPU
OPTIMIZED_IMG_SIZE = 224  # Reduced from 448 for 4x faster preprocessing
val_tfms = A.Compose([
    A.LongestMaxSize(max_size=OPTIMIZED_IMG_SIZE),
    A.PadIfNeeded(OPTIMIZED_IMG_SIZE, OPTIMIZED_IMG_SIZE, border_mode=cv2.BORDER_REFLECT_101),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Global model cache with thread pool for CPU optimization
pro_model = None
clinical_model = None
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)  # Single worker for model inference

@lru_cache(maxsize=2)
def load_model_cached(model_path: str, model_type: str):
    """Cached model loading with CPU optimizations"""
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        logger.info(f"Loading {model_type} model from {model_path}")
        model = timm.create_model('densenet121', pretrained=False, num_classes=1)
        
        # Load with error handling
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        
        # CPU optimizations with error handling
        try:
            model = torch.jit.optimize_for_inference(torch.jit.script(model))
            logger.info(f"{model_type} model loaded and JIT optimized on {DEVICE}")
        except Exception as jit_error:
            logger.warning(f"JIT optimization failed for {model_type} model: {jit_error}")
            logger.info(f"{model_type} model loaded without JIT optimization on {DEVICE}")
            
        return model
    except Exception as e:
        logger.error(f"Error loading {model_type} model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load AI {model_type} model: {str(e)}")

def get_model(is_pro: bool = False):
    """Get cached model instance"""
    try:
        if is_pro:
            return load_model_cached(PRO_MODEL_PATH, "pro")
        else:
            return load_model_cached(CLINICAL_MODEL_PATH, "clinical")
    except Exception as e:
        logger.error(f"Failed to get model (is_pro={is_pro}): {e}")
        raise

def preprocess_image(image_bytes: bytes):
    """Optimized image preprocessing for CPU inference"""
    try:
        # Convert bytes to PIL Image with size optimization
        image = Image.open(image_bytes).convert('RGB')
        
        # Pre-resize if image is very large to speed up processing
        if max(image.size) > 1024:
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Apply optimized transforms
        transformed = val_tfms(image=img_array)
        img_tensor = transformed["image"].unsqueeze(0)  # Add batch dimension
        
        return img_tensor.to(DEVICE)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def predict_image_sync(img_tensor, is_pro: bool = False):
    """Synchronous CPU-optimized inference"""
    try:
        model = get_model(is_pro=is_pro)
        
        with torch.inference_mode():
            # CPU inference without autocast
            logits = model(img_tensor).view(-1)
            probability = torch.sigmoid(logits).item()
            is_malignant = probability > 0.5
            classification = "malignant" if is_malignant else "benign"
            
            return {
                "classification": classification,
                "probability": {
                    "malignant": float(probability),
                    "benign": float(1 - probability)
                },
                "confidence": float(max(probability, 1 - probability))
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

async def predict_image(img_tensor, is_pro: bool = False):
    """Async wrapper for CPU inference in thread pool"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, predict_image_sync, img_tensor, is_pro)

def save_uploaded_image(file_content: bytes, filename: str) -> str:
    """Save uploaded image to disk and return the URL path for frontend access"""
    try:
        # Generate unique filename to avoid conflicts
        file_extension = os.path.splitext(filename)[1].lower()
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Save file to disk
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        # Return URL path that can be accessed via HTTP (relative to API base)
        # This will be served by FastAPI static file mounting
        url_path = f"/uploads/{unique_filename}"
        return url_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving image: {str(e)}")

@router.post("/classify")
async def classify_mole(
    file: UploadFile = File(...),
    skinTone: Optional[str] = Form(None),
    bodyPart: Optional[str] = Form(None),
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Classify a mole image as benign or malignant
    
    - **file**: Image file (JPEG, PNG, etc.)
    - **skinTone**: Optional skin tone classification
    - **bodyPart**: Optional body part location
    - Returns classification result with probability scores
    - Requires authentication
    """
    
    # Validate file type - allow common image extensions even if MIME type is not detected
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
    file_extension = os.path.splitext(file.filename.lower())[1] if file.filename else ''
    
    if not (file.content_type and file.content_type.startswith('image/')) and file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"File must be an image. Supported formats: {', '.join(allowed_extensions)}")
    
    try:
        # Read image file
        image_bytes = await file.read()
        
        # Save image to disk - returns URL path for frontend access
        image_url_path = save_uploaded_image(image_bytes, file.filename)
        # Get the actual file system path for cleanup if needed
        unique_filename = os.path.basename(image_url_path)
        file_system_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Preprocess image for prediction
        img_tensor = preprocess_image(io.BytesIO(image_bytes))
        
        # Run async prediction with user's pro status
        result = await predict_image(img_tensor, is_pro=current_user.is_pro)
        
        # Save picture and diagnosis records to database with proper error handling
        try:
            logger.info(f"Saving classification for user {current_user.id}")
            
            # Save picture record to database with URL path, skin tone, and body part
            picture = Picture(
                user_id=current_user.id,
                image_path=image_url_path,  # Store URL path for frontend access
                filename=file.filename,
                skin_tone=skinTone,
                body_part_location=bodyPart
            )
            db.add(picture)
            db.flush()  # Flush to get the picture ID without committing
            logger.info(f"Picture record created with ID: {picture.id}")
            
            # Save diagnosis record to database
            diagnosis = Diagnosis(
                picture_id=picture.id,
                user_id=current_user.id,
                diagnosis=result["classification"],
                malignant_probability=result["probability"]["malignant"],
                benign_probability=result["probability"]["benign"],
                confidence=result["confidence"],
                date_of_diagnosis=date.today()
            )
            db.add(diagnosis)
            db.commit()  # Commit both records together
            db.refresh(picture)
            db.refresh(diagnosis)
            logger.info(f"Diagnosis record created with ID: {diagnosis.id}")
            
        except Exception as db_error:
            # Rollback the transaction if database operations fail
            logger.error(f"Database error during classification save: {db_error}")
            db.rollback()
            # Clean up saved image file using file system path
            if os.path.exists(file_system_path):
                os.remove(file_system_path)
            raise HTTPException(
                status_code=500, 
                detail=f"Database error during classification save: {str(db_error)}"
            )
        
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "image_url": image_url_path,  # URL path for frontend to display the image
            "skin_tone": skinTone,
            "body_part": bodyPart,
            "user_id": current_user.id,
            "picture_id": picture.id,
            "diagnosis_id": diagnosis.id,
            "result": result
        })
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up saved image if any operation fails
        if 'file_system_path' in locals() and os.path.exists(file_system_path):
            try:
                os.remove(file_system_path)
            except OSError:
                pass  # Ignore cleanup errors
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@router.get("/model-info")
def get_model_info():
    """Get information about the loaded models"""
    try:
        pro_model_exists = os.path.exists(PRO_MODEL_PATH)
        clinical_model_exists = os.path.exists(CLINICAL_MODEL_PATH)
        return {
            "models": {
                "pro": {
                    "path": PRO_MODEL_PATH,
                    "exists": pro_model_exists,
                    "description": "High-accuracy model for pro users"
                },
                "clinical": {
                    "path": CLINICAL_MODEL_PATH,
                    "exists": clinical_model_exists,
                    "description": "Clinical-grade model for regular users"
                }
            },
            "device": str(DEVICE),
            "image_size": IMG_SIZE,
            "model_architecture": "DenseNet-121"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")