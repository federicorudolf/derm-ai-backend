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
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from datetime import date
import logging
from auth import get_current_user
from models import User as UserModel, Picture, Diagnosis
from database import get_db

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

# Model configuration
MODEL_PATH = "./checkpoints/derm_densenet121_best.pth"
IMG_SIZE = 448
UPLOAD_DIR = "./uploads"

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Device setup
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Image preprocessing transforms (same as validation transforms from training)
val_tfms = A.Compose([
    A.LongestMaxSize(max_size=IMG_SIZE),
    A.PadIfNeeded(IMG_SIZE, IMG_SIZE, border_mode=cv2.BORDER_REFLECT_101),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Load model once at startup
model = None

def load_model():
    global model
    if model is None:
        try:
            # Create the same model architecture as in training
            model = timm.create_model('densenet121', pretrained=True, num_classes=1)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            model = model.to(torch.float32)
            print(f"Model loaded successfully on {DEVICE}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise HTTPException(status_code=500, detail="Failed to load AI model")
    return model

def preprocess_image(image_bytes: bytes):
    """Preprocess uploaded image for model inference"""
    try:
        # Convert bytes to PIL Image
        image = Image.open(image_bytes).convert('RGB')
        # Convert to numpy array
        img_array = np.array(image)
        
        # Apply the same transforms as used during validation
        transformed = val_tfms(image=img_array)
        img_tensor = transformed["image"].unsqueeze(0)  # Add batch dimension
        
        return img_tensor.to(DEVICE)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def predict_image(img_tensor):
    """Run inference on preprocessed image"""
    try:
        model = load_model()
        
        with torch.inference_mode():
            # Use autocast for inference
            with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type != 'cpu')):
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
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Classify a mole image as benign or malignant
    
    - **file**: Image file (JPEG, PNG, etc.)
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
        
        # Run prediction
        result = predict_image(img_tensor)
        
        # Save picture and diagnosis records to database with proper error handling
        try:
            logger.info(f"Saving classification for user {current_user.id}")
            
            # Save picture record to database with URL path
            picture = Picture(
                user_id=current_user.id,
                image_path=image_url_path,  # Store URL path for frontend access
                filename=file.filename
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
    """Get information about the loaded model"""
    try:
        model_exists = os.path.exists(MODEL_PATH)
        return {
            "model_path": MODEL_PATH,
            "model_exists": model_exists,
            "device": str(DEVICE),
            "image_size": IMG_SIZE,
            "model_architecture": "DenseNet-121"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")