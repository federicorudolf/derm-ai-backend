from pydantic import BaseModel, EmailStr
from datetime import date, datetime
from typing import Optional, List

class UserBase(BaseModel):
    name: str
    email: EmailStr
    bio: Optional[str] = None
    d_o_b: Optional[date] = None
    country: Optional[str] = None

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class User(UserBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class Picture(BaseModel):
    id: int
    user_id: int
    body_part_location: Optional[str] = None
    image_path: str
    filename: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class PaginatedPicturesResponse(BaseModel):
    pictures: List[Picture]
    total: int
    page: int
    size: int
    total_pages: int

class SessionCreate(BaseModel):
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None

class Session(BaseModel):
    id: int
    user_id: int
    is_active: bool
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    expires_at: datetime
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class TokenWithSession(BaseModel):
    access_token: str
    token_type: str
    session_id: int
    expires_at: datetime