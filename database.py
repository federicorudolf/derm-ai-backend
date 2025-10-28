from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import QueuePool
import os
from dotenv import load_dotenv
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://username:password@localhost/dermAI")

# Enhanced engine configuration to handle SSL and connection issues
engine = create_engine(
    DATABASE_URL,
    # Connection pool settings
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # This will test connections before use
    pool_recycle=3600,   # Recycle connections every hour
    
    # SSL and connection settings
    connect_args={
        "sslmode": "prefer",  # Use SSL if available, but don't require it
        "connect_timeout": 10,
        "application_name": "dermAI_backend",
    },
    
    # Additional settings for reliability
    echo=False,  # Set to True for SQL query logging
    future=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def test_connection():
    """Test database connection"""
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False