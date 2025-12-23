from datetime import timedelta, datetime
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm

from database import get_db
from models import User as UserModel
from schemas import (
    UserCreate, User, Token, UserLogin, TokenWithSession,
    ForgotPasswordRequest, ForgotPasswordResponse,
    ResetPasswordRequest, ResetPasswordResponse,
    VerifyResetTokenResponse
)
from auth import (
    authenticate_user,
    create_access_token,
    get_password_hash,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    get_current_user,
    create_session,
    invalidate_session,
    invalidate_user_sessions,
    generate_reset_token,
    create_password_reset_token,
    validate_reset_token,
    mark_reset_token_as_used,
    invalidate_all_user_reset_tokens
)
from services.email_service import send_password_reset_email
import logging
import asyncio

router = APIRouter()
logger = logging.getLogger(__name__)

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
        invalidate_user_sessions(db, current_user.id)
        from models import Diagnosis, Picture
        pictures = db.query(Picture).filter(Picture.user_id == current_user.id).all()
        db.query(Diagnosis).filter(Diagnosis.user_id == current_user.id).delete()
        db.query(Picture).filter(Picture.user_id == current_user.id).delete()

        from models import Session as SessionModel
        db.query(SessionModel).filter(SessionModel.user_id == current_user.id).delete()
        db.delete(current_user)
        db.commit()

        import os
        deleted_files = 0
        failed_files = 0

        for picture in pictures:
            try:
                # Extract filename from URL path (e.g., "/uploads/images/abc123.jpg" -> "abc123.jpg")
                if picture.image_path:
                    filename = os.path.basename(picture.image_path)
                    # Determine upload directory based on environment
                    if os.environ.get("RAILWAY_ENVIRONMENT"):
                        upload_dir = "/uploads/images"
                    else:
                        upload_dir = os.path.join(os.getcwd(), "uploads", "images")

                    file_path = os.path.join(upload_dir, filename)

                    if os.path.exists(file_path):
                        os.remove(file_path)
                        deleted_files += 1
            except Exception as file_error:
                # Log but don't fail the entire operation if file deletion fails
                failed_files += 1
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to delete image file {picture.image_path}: {file_error}")

        return {
            "message": "Account successfully deleted",
            "files_deleted": deleted_files,
            "files_failed": failed_files
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete account"
        )

@router.post("/forgot-password", response_model=ForgotPasswordResponse)
async def forgot_password(
    request_data: ForgotPasswordRequest,
    db: Session = Depends(get_db)
):
    """
    Request a password reset email

    Security: Always returns success message even if email doesn't exist
    to prevent email enumeration attacks
    """
    try:
        # Look up user by email
        user = db.query(UserModel).filter(UserModel.email == request_data.email).first()

        if user:
            # Invalidate any existing reset tokens for this user
            invalidate_all_user_reset_tokens(db, user.id)

            # Generate new reset token
            reset_token = generate_reset_token()

            # Store hashed token in database
            create_password_reset_token(db, user.id, reset_token)

            # Send email asynchronously (fire and forget for performance)
            asyncio.create_task(
                send_password_reset_email(
                    recipient_email=user.email,
                    recipient_name=user.name,
                    reset_token=reset_token
                )
            )

            logger.info(f"Password reset requested for user {user.id}")
        else:
            # Log for security monitoring but don't reveal to user
            logger.info(f"Password reset requested for non-existent email: {request_data.email}")

        # ALWAYS return generic success message (security best practice)
        return {
            "message": "Si el correo electrónico existe en nuestro sistema, recibirás un enlace para restablecer tu contraseña."
        }

    except Exception as e:
        logger.error(f"Error in forgot_password endpoint: {e}")
        # Still return generic message even on error
        return {
            "message": "Si el correo electrónico existe en nuestro sistema, recibirás un enlace para restablecer tu contraseña."
        }

@router.post("/reset-password", response_model=ResetPasswordResponse)
def reset_password(
    request_data: ResetPasswordRequest,
    db: Session = Depends(get_db)
):
    """
    Reset password using a valid reset token

    Security features:
    - Validates token hasn't expired or been used
    - Invalidates all user sessions after password change
    - Marks token as used to prevent reuse
    """
    # Validate the reset token
    reset_token = validate_reset_token(db, request_data.token)

    if not reset_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token de restablecimiento inválido o expirado"
        )

    try:
        # Get the user
        user = db.query(UserModel).filter(UserModel.id == reset_token.user_id).first()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Usuario no encontrado"
            )

        # Update password
        user.password_hash = get_password_hash(request_data.new_password)
        user.updated_at = datetime.utcnow()

        # Mark reset token as used
        mark_reset_token_as_used(db, reset_token.id)

        # SECURITY: Invalidate all existing sessions (required by user requirements)
        invalidated_count = invalidate_user_sessions(db, user.id)

        db.commit()

        logger.info(
            f"Password reset successful for user {user.id}. "
            f"Invalidated {invalidated_count} sessions."
        )

        return {
            "message": "Contraseña restablecida exitosamente. Por favor inicia sesión con tu nueva contraseña."
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error resetting password: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al restablecer la contraseña"
        )

@router.get("/verify-reset-token", response_model=VerifyResetTokenResponse)
def verify_reset_token_endpoint(
    token: str,
    db: Session = Depends(get_db)
):
    """
    Optional endpoint to verify if a reset token is valid
    Useful for frontend to show appropriate UI before password submission
    """
    reset_token = validate_reset_token(db, token)

    if reset_token:
        return {
            "valid": True,
            "message": "Token válido"
        }
    else:
        return {
            "valid": False,
            "message": "Token inválido o expirado"
        }