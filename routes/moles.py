from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import desc
from typing import Optional
import math

from database import get_db
from models import Mole as MoleModel, Picture as PictureModel, Diagnosis as DiagnosisModel, User as UserModel
from schemas import MoleCreate, MoleUpdate, Mole, MoleWithSummary, MoleDetail, PaginatedMolesResponse
from auth import get_current_user

router = APIRouter()

@router.get("/moles", response_model=PaginatedMolesResponse)
async def get_user_moles(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(10, ge=1, le=100, description="Number of items per page"),
    include_archived: bool = Query(False, description="Include archived moles"),
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get paginated list of moles for the authenticated user with summary info.
    Returns moles sorted by most recent picture first.
    """
    offset = (page - 1) * size

    # Base query
    query = db.query(MoleModel).filter(MoleModel.user_id == current_user.id)

    # Filter archived if needed
    if not include_archived:
        query = query.filter(MoleModel.is_archived == False)

    # Get total count
    total = query.count()

    # Get moles with eager loading
    moles = query.options(
        joinedload(MoleModel.pictures).joinedload(PictureModel.diagnoses)
    ).offset(offset).limit(size).all()

    # Build summary response
    moles_with_summary = []
    for mole in moles:
        # Get picture count
        picture_count = len(mole.pictures)

        # Get latest picture (pictures are already ordered by created_at desc)
        latest_picture = mole.pictures[0] if mole.pictures else None

        # Get latest diagnosis from latest picture
        latest_diagnosis = None
        if latest_picture and latest_picture.diagnoses:
            latest_diagnosis = latest_picture.diagnoses[0]

        mole_summary = MoleWithSummary(
            id=mole.id,
            user_id=mole.user_id,
            name=mole.name,
            body_part_location=mole.body_part_location,
            notes=mole.notes,
            is_archived=mole.is_archived,
            created_at=mole.created_at,
            updated_at=mole.updated_at,
            picture_count=picture_count,
            latest_picture=latest_picture,
            latest_diagnosis=latest_diagnosis
        )
        moles_with_summary.append(mole_summary)

    # Sort by latest picture date
    moles_with_summary.sort(
        key=lambda x: x.latest_picture.created_at if x.latest_picture else x.created_at,
        reverse=True
    )

    total_pages = math.ceil(total / size) if total > 0 else 0

    return PaginatedMolesResponse(
        moles=moles_with_summary,
        total=total,
        page=page,
        size=size,
        total_pages=total_pages
    )

@router.post("/moles", response_model=Mole)
async def create_mole(
    mole_data: MoleCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new mole for the authenticated user.
    """
    mole = MoleModel(
        user_id=current_user.id,
        name=mole_data.name,
        body_part_location=mole_data.body_part_location,
        notes=mole_data.notes
    )
    db.add(mole)
    db.commit()
    db.refresh(mole)
    return mole

@router.get("/moles/{mole_id}", response_model=MoleDetail)
async def get_mole_detail(
    mole_id: int,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific mole including all pictures and diagnoses.
    """
    mole = db.query(MoleModel).filter(
        MoleModel.id == mole_id,
        MoleModel.user_id == current_user.id
    ).options(
        joinedload(MoleModel.pictures).joinedload(PictureModel.diagnoses)
    ).first()

    if not mole:
        raise HTTPException(status_code=404, detail="Mole not found")

    return mole

@router.put("/moles/{mole_id}", response_model=Mole)
async def update_mole(
    mole_id: int,
    mole_data: MoleUpdate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update mole information (name, notes, etc.)
    """
    mole = db.query(MoleModel).filter(
        MoleModel.id == mole_id,
        MoleModel.user_id == current_user.id
    ).first()

    if not mole:
        raise HTTPException(status_code=404, detail="Mole not found")

    # Update fields
    update_data = mole_data.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(mole, key, value)

    db.commit()
    db.refresh(mole)
    return mole

@router.delete("/moles/{mole_id}")
async def archive_mole(
    mole_id: int,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Archive a mole (soft delete)
    """
    mole = db.query(MoleModel).filter(
        MoleModel.id == mole_id,
        MoleModel.user_id == current_user.id
    ).first()

    if not mole:
        raise HTTPException(status_code=404, detail="Mole not found")

    mole.is_archived = True
    db.commit()

    return {"message": "Mole archived successfully"}
