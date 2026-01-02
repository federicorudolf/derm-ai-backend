from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session as DBSession
from database import get_db
from models import User, Session as SessionModel, PasswordResetToken
from dotenv import load_dotenv
import os
import hashlib
import secrets

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "360"))
PASSWORD_RESET_TOKEN_EXPIRE_MINUTES = 60  # 1 hour

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def authenticate_user(db: DBSession, email: str, password: str):
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            return False
        if not verify_password(password, user.password_hash):
            return False
        return user
    except Exception as e:
        # Log the error and return False to indicate authentication failure
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Database error during authentication: {e}")
        return False

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: DBSession = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token = credentials.credentials
    
    try:
        # First validate the JWT token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    # Then validate the session
    session = validate_session(db, token)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session invalid or expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        user = db.query(User).filter(User.email == email).first()
        if user is None:
            raise credentials_exception
        return user
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Database error during user lookup: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service temporarily unavailable"
        )

def create_token_hash(token: str) -> str:
    """Create a hash of the token for secure storage"""
    return hashlib.sha256(token.encode()).hexdigest()

def create_session(db: DBSession, user_id: int, token: str, expires_at: datetime, user_agent: str = None, ip_address: str = None) -> SessionModel:
    """Create a new session record"""
    token_hash = create_token_hash(token)
    
    session = SessionModel(
        user_id=user_id,
        token_hash=token_hash,
        is_active=True,
        user_agent=user_agent,
        ip_address=ip_address,
        expires_at=expires_at
    )
    
    db.add(session)
    db.commit()
    db.refresh(session)
    return session

def invalidate_session(db: DBSession, session_id: int) -> bool:
    """Invalidate a session by setting is_active to False"""
    try:
        session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
        if session:
            session.is_active = False
            db.commit()
            return True
        return False
    except Exception:
        return False

def invalidate_user_sessions(db: DBSession, user_id: int) -> int:
    """Invalidate all sessions for a user. Returns count of invalidated sessions."""
    try:
        count = db.query(SessionModel).filter(
            SessionModel.user_id == user_id,
            SessionModel.is_active == True
        ).update({"is_active": False})
        db.commit()
        return count
    except Exception:
        return 0

def validate_session(db: DBSession, token: str) -> SessionModel:
    """Validate if a session is active and not expired"""
    token_hash = create_token_hash(token)
    
    session = db.query(SessionModel).filter(
        SessionModel.token_hash == token_hash,
        SessionModel.is_active == True,
        SessionModel.expires_at > datetime.utcnow()
    ).first()
    
    return session

def cleanup_expired_sessions(db: DBSession) -> int:
    """Clean up expired sessions. Returns count of cleaned sessions."""
    try:
        count = db.query(SessionModel).filter(
            SessionModel.expires_at <= datetime.utcnow()
        ).update({"is_active": False})
        db.commit()
        return count
    except Exception:
        return 0

# Password Reset Token Functions

def generate_reset_token() -> str:
    """Generate a secure random token for password reset (32 bytes = 64 hex chars)"""
    return secrets.token_urlsafe(32)

def create_password_reset_token(
    db: DBSession,
    user_id: int,
    token: str
) -> PasswordResetToken:
    """
    Create a password reset token record

    Args:
        db: Database session
        user_id: User ID
        token: Plain token (will be hashed before storage)

    Returns:
        PasswordResetToken model instance
    """
    token_hash = create_token_hash(token)
    expires_at = datetime.utcnow() + timedelta(minutes=PASSWORD_RESET_TOKEN_EXPIRE_MINUTES)

    reset_token = PasswordResetToken(
        user_id=user_id,
        token_hash=token_hash,
        expires_at=expires_at,
        is_used=False
    )

    db.add(reset_token)
    db.commit()
    db.refresh(reset_token)

    return reset_token

def validate_reset_token(db: DBSession, token: str) -> Optional[PasswordResetToken]:
    """
    Validate a password reset token

    Args:
        db: Database session
        token: Plain token to validate

    Returns:
        PasswordResetToken if valid, None otherwise
    """
    token_hash = create_token_hash(token)

    reset_token = db.query(PasswordResetToken).filter(
        PasswordResetToken.token_hash == token_hash,
        PasswordResetToken.is_used == False,
        PasswordResetToken.expires_at > datetime.utcnow()
    ).first()

    return reset_token

def mark_reset_token_as_used(db: DBSession, token_id: int) -> bool:
    """Mark a reset token as used to prevent reuse"""
    try:
        reset_token = db.query(PasswordResetToken).filter(
            PasswordResetToken.id == token_id
        ).first()

        if reset_token:
            reset_token.is_used = True
            db.commit()
            return True
        return False
    except Exception:
        return False

def invalidate_all_user_reset_tokens(db: DBSession, user_id: int) -> int:
    """
    Invalidate all reset tokens for a user (for security)
    Returns count of invalidated tokens
    """
    try:
        count = db.query(PasswordResetToken).filter(
            PasswordResetToken.user_id == user_id,
            PasswordResetToken.is_used == False
        ).update({"is_used": True})
        db.commit()
        return count
    except Exception:
        return 0

def cleanup_expired_reset_tokens(db: DBSession) -> int:
    """
    Clean up expired or used reset tokens
    Returns count of deleted tokens
    """
    try:
        count = db.query(PasswordResetToken).filter(
            (PasswordResetToken.expires_at <= datetime.utcnow()) |
            (PasswordResetToken.is_used == True)
        ).delete()
        db.commit()
        return count
    except Exception:
        return 0