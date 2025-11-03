from sqlalchemy import Column, Integer, String, DateTime, Date, ForeignKey, Float, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    bio = Column(String)
    d_o_b = Column(Date)
    country = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    pictures = relationship("Picture", back_populates="user")
    diagnoses = relationship("Diagnosis", back_populates="user")
    sessions = relationship("Session", back_populates="user")

class Picture(Base):
    __tablename__ = "pictures"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    body_part_location = Column(String, nullable=True)
    image_path = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User", back_populates="pictures")
    diagnoses = relationship("Diagnosis", back_populates="picture")

class Diagnosis(Base):
    __tablename__ = "diagnoses"
    
    id = Column(Integer, primary_key=True, index=True)
    picture_id = Column(Integer, ForeignKey("pictures.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    diagnosis = Column(String, nullable=False)
    malignant_probability = Column(Float, nullable=False)
    benign_probability = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    date_of_diagnosis = Column(Date, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    picture = relationship("Picture", back_populates="diagnoses")
    user = relationship("User", back_populates="diagnoses")

class Session(Base):
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    token_hash = Column(String, nullable=False, index=True)
    is_active = Column(Boolean, default=True, nullable=False)
    user_agent = Column(Text)
    ip_address = Column(String)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User", back_populates="sessions")