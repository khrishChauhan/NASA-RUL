import os
from pathlib import Path
from pydantic import BaseModel

class Settings(BaseModel):
    PROJECT_NAME: str = "NASA-RUL — Predictive Maintenance Intelligence"
    API_V1_STR: str = "/api"
    
    # Resolve project root relative to this file: backend/app/core/config.py -> 3 levels up
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent
    
    DATA_DIR: Path = PROJECT_ROOT / "data" / "raw" / "cmapss"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    APP_DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
    
    # Model Artifact paths
    LSTM_WEIGHTS_PATH: Path = MODELS_DIR / "lstm_fd001.pth"
    LSTM_CONFIG_PATH: Path = MODELS_DIR / "lstm_fd001_config.json"
    TRANSFORMER_WEIGHTS_PATH: Path = MODELS_DIR / "transformer_fd001.pth"
    SCALER_PATH: Path = MODELS_DIR / "scaler_fd001.joblib"
    
    # Bundled deployment sample asset (for production environments where raw dataset is excluded)
    SAMPLE_WINDOWS_PATH: Path = APP_DATA_DIR / "sample_fd001_windows.json"
    
    # Dataset files (local development)
    TRAIN_DATA_PATH: Path = DATA_DIR / "train_FD001.txt"
    TEST_DATA_PATH: Path = DATA_DIR / "test_FD001.txt"
    RUL_DATA_PATH: Path = DATA_DIR / "RUL_FD001.txt"
    
    SEQUENCE_LENGTH: int = 30
    DEFAULT_MODEL: str = "ImprovedRULLSTM"
    
    # CORS: Allow all origins by default for public inference, or comma-separated FRONTEND_ORIGIN env var
    FRONTEND_ORIGIN: str = os.getenv("FRONTEND_ORIGIN", "*")
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ]

settings = Settings()

