from datetime import timedelta, datetime
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm

from database import get_db
from models import User as UserModel
from schemas import UserCreate, User, Token, UserLogin, TokenWithSession
from auth import (
    authenticate_user, 
    create_access_token, 
    get_password_hash, 
    ACCESS_TOKEN_EXPIRE_MINUTES,
    get_current_user,
    create_session,
    invalidate_session,
    invalidate_user_sessions
)

router = APIRouter()

@router.post("/signup", response_model=TokenWithSession)
def register_user(user: UserCreate, request: Request, db: Session = Depends(get_db)):
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
    
    # Create access token and session for the newly registered user
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": db_user.email}, expires_delta=access_token_expires
    )
    
    # Create session
    expires_at = datetime.utcnow() + access_token_expires
    user_agent = request.headers.get("user-agent")
    ip_address = request.client.host if request.client else None
    
    session = create_session(
        db=db,
        user_id=db_user.id,
        token=access_token,
        expires_at=expires_at,
        user_agent=user_agent,
        ip_address=ip_address
    )
    
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "session_id": session.id,
        "expires_at": expires_at
    }

@router.post("/login", response_model=TokenWithSession)
def login_user(user_credentials: UserLogin, request: Request, db: Session = Depends(get_db)):
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
    
    # Create session
    expires_at = datetime.utcnow() + access_token_expires
    user_agent = request.headers.get("user-agent")
    ip_address = request.client.host if request.client else None
    
    session = create_session(
        db=db,
        user_id=user.id,
        token=access_token,
        expires_at=expires_at,
        user_agent=user_agent,
        ip_address=ip_address
    )
    
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "session_id": session.id,
        "expires_at": expires_at
    }

@router.get("/me", response_model=User)
def read_users_me(current_user: UserModel = Depends(get_current_user)):
    return current_user

@router.post("/logout")
def logout_user(request: Request, current_user: UserModel = Depends(get_current_user), db: Session = Depends(get_db)):
    """Logout current session"""
    # Get token from authorization header
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header"
        )
    
    token = auth_header.split(" ")[1]
    
    # Find and invalidate the session
    from auth import create_token_hash
    from models import Session as SessionModel
    token_hash = create_token_hash(token)
    
    user_session = db.query(SessionModel).filter(
        SessionModel.user_id == current_user.id,
        SessionModel.token_hash == token_hash,
        SessionModel.is_active == True
    ).first()
    
    if user_session:
        invalidate_session(db, user_session.id)
        return {"message": "Successfully logged out"}
    else:
        return {"message": "Session not found or already logged out"}

@router.post("/logout-all")
def logout_all_sessions(current_user: UserModel = Depends(get_current_user), db: Session = Depends(get_db)):
    """Logout all sessions for current user"""
    count = invalidate_user_sessions(db, current_user.id)
    return {"message": f"Successfully logged out from {count} sessions"}

@router.delete("/users/me")
def delete_user_account(current_user: UserModel = Depends(get_current_user), db: Session = Depends(get_db)):
    """Delete the current user's account and all associated data"""
    try:
        # First, invalidate all user sessions
        invalidate_user_sessions(db, current_user.id)
        
        # Delete all related data in order (due to foreign key constraints)
        # Delete diagnoses first
        from models import Diagnosis, Picture
        db.query(Diagnosis).filter(Diagnosis.user_id == current_user.id).delete()
        
        # Delete pictures
        db.query(Picture).filter(Picture.user_id == current_user.id).delete()
        
        # Delete sessions (should already be invalidated but let's remove them)
        from models import Session as SessionModel
        db.query(SessionModel).filter(SessionModel.user_id == current_user.id).delete()
        
        # Finally delete the user
        db.delete(current_user)
        db.commit()
        
        return {"message": "Account successfully deleted"}
    
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete account"
        )