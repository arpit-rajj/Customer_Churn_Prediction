from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    DATABASE_URL: str
    LOCAL_DATABASE_URL: Optional[str] = None
    MODEL_DIR: str = "models"
    MODEL_VERSION: str = "rf-v1"

    class Config:
        env_file = ".env"

settings = Settings()
