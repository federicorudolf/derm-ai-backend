from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm

from database import get_db
from models import User as UserModel
from schemas import UserCreate, User, Token, UserLogin
from auth import (
    authenticate_user, 
    create_access_token, 
    get_password_hash, 
    ACCESS_TOKEN_EXPIRE_MINUTES,
    get_current_user
)

router = APIRouter()

@router.post("/signup", response_model=User)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(UserModel).filter(UserModel.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    db_user = UserModel(
        name=user.name,
        email=user.email,
        password_hash=hashed_password,
        bio=user.bio,
        d_o_b=user.d_o_b,
        country=user.country
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@router.post("/login", response_model=Token)
def login_user(user_credentials: UserLogin, db: Session = Depends(get_db)):
    user = authenticate_user(db, user_credentials.email, user_credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=User)
def read_users_me(current_user: UserModel = Depends(get_current_user)):
    return current_user