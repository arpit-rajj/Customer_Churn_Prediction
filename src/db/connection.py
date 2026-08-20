from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from src.config import settings

# Depending on if we are running locally (e.g. scripts) or in docker, we might use different DB URLs.
# For simplicity, we try to use LOCAL_DATABASE_URL if available, but default to DATABASE_URL.
# If running scripts from host, LOCAL_DATABASE_URL should point to localhost.
DB_URL = settings.LOCAL_DATABASE_URL if settings.LOCAL_DATABASE_URL else settings.DATABASE_URL

engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
