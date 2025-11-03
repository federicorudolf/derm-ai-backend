from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional
import math

from database import get_db
from models import Picture as PictureModel, User as UserModel
from schemas import PaginatedPicturesResponse
from auth import get_current_user

router = APIRouter()

@router.get("/images", response_model=PaginatedPicturesResponse)
async def get_user_images(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(10, ge=1, le=100, description="Number of items per page"),
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get paginated list of images for the authenticated user
    """
    # Calculate offset for pagination
    offset = (page - 1) * size
    
    # Get total count of user's pictures
    total = db.query(func.count(PictureModel.id)).filter(
        PictureModel.user_id == current_user.id
    ).scalar()
    
    # Get paginated pictures
    pictures = db.query(PictureModel).filter(
        PictureModel.user_id == current_user.id
    ).order_by(PictureModel.created_at.desc()).offset(offset).limit(size).all()
    
    # Calculate total pages
    total_pages = math.ceil(total / size) if total > 0 else 0
    
    return PaginatedPicturesResponse(
        pictures=pictures,
        total=total,
        page=page,
        size=size,
        total_pages=total_pages
    )