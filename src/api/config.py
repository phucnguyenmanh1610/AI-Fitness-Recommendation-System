"""
Configuration settings for the API
"""
import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    API_TITLE: str = "AI Fitness & Health Recommendation System"
    API_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Model Paths
    MODEL_DIR: str = "models"
    CALORIE_MODEL_PATH: str = os.path.join(MODEL_DIR, "calorie_model.pkl")
    BMI_MODEL_PATH: str = os.path.join(MODEL_DIR, "bmi_model.pkl")
    MAPPINGS_PATH: str = os.path.join(MODEL_DIR, "categorical_mappings.pkl")
    
    # Data Paths
    DATA_DIR: str = "data"
    RAW_DATA_PATH: str = os.path.join(DATA_DIR, "raw", "fitness.csv")
    WORKOUT_ITEMS_PATH: str = os.path.join(DATA_DIR, "train", "items.csv")
    MEAL_DATABASE_PATH: str = os.path.join(DATA_DIR, "train", "meals.csv")
    
    # Database Settings
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL", "sqlite:///./fitness.db")
    
    # Performance Settings
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    
    # CORS Settings
    CORS_ORIGINS: list = ["*"]  # In production, specify actual origins
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

