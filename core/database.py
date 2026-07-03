from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, declarative_base
from core.config import settings

# Create synchronous engine for migrations and general control-plane tasks
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """
    Dependency helper to yield database session and close it after request finishes.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
