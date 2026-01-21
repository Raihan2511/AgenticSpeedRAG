import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Project Info
    PROJECT_NAME: str = "Agentic SpeedRAG"
    VERSION: str = "1.0.0"
    
    # Database Configs
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    POSTGRES_DSN: str = os.getenv("POSTGRES_DSN", "postgresql://admin:rootpassword@localhost:5432/speedrag_db")
    
    # LLM Configs (Krutrim)
    # We leave these empty by default to force you to set them in .env
    KRUTRIM_API_KEY: str
    KRUTRIM_MODEL: str = "Krutrim-spectre-v2" # check specific model name
    KRUTRIM_BASE_URL: str = "https://cloud.olakrutrim.com/v1" # verify this endpoint
    
    class Config:
        env_file = ".env"

# Initialize single instance
settings = Settings()