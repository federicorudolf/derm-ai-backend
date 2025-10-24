from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import Base
from database import engine
from routes import auth, classification

Base.metadata.create_all(bind=engine)

app = FastAPI(title="DermAI Backend", version="1.0.0")

import os
from dotenv import load_dotenv

load_dotenv()

# Get allowed origins from environment or use localhost for development
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(auth.router, prefix="/auth", tags=["authentication"])
app.include_router(classification.router, prefix="/api", tags=["classification"])

@app.get("/")
def read_root():
    return {"message": "Welcome to DermAI Backend API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}