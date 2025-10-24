from pydantic import BaseModel, EmailStr
from datetime import date, datetime
from typing import Optional

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