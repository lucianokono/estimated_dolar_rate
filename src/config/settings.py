# Configurações do Sistema

import os
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Configurações principais do sistema"""
    
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/dollar_estimation"
    REDIS_URL: str = "redis://localhost:6379"
    
    # API Keys (não necessárias para APIs brasileiras)
    # ALPHA_VANTAGE_API_KEY: str = ""  # removido - usando APIs brasileiras
    # NEWS_API_KEY: str = ""  # removido - usando scraping direto
    
    # Model Configuration
    MODEL_VERSION: str = "v1.0"
    MODEL_RETRAIN_THRESHOLD: float = 0.1
    PREDICTION_CACHE_TTL: int = 3600
    
    # Data Sources - APIs brasileiras
    NEWS_SOURCES: List[str] = [
        "https://g1.globo.com/economia/",
        "https://www.valor.com.br/",
        "https://www.infomoney.com.br/",
        "https://www.estadao.com.br/economia/",
        "https://www.folha.uol.com.br/mercado/"
    ]
    
    # APIs brasileiras
    BCB_API_BASE: str = "https://api.bcb.gov.br/dados/serie/bcdata.sgs"
    B3_API_BASE: str = "https://www2.bmf.com.br"
    
    # NLP Configuration
    MAX_TEXT_LENGTH: int = 512
    BATCH_SIZE: int = 32
    SENTIMENT_MODEL: str = "neuralmind/bert-base-portuguese-cased"
    
    # Feature Engineering
    LOOKBACK_DAYS: int = 30
    TECHNICAL_INDICATORS: List[str] = [
        "sma_5", "sma_20", "ema_12", "ema_26",
        "rsi", "macd", "bollinger_upper", "bollinger_lower"
    ]
    
    # Model Training
    TRAIN_TEST_SPLIT: float = 0.8
    VALIDATION_SPLIT: float = 0.2
    CROSS_VALIDATION_FOLDS: int = 5
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/system.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Instância global das configurações
settings = Settings()

# Diretórios do projeto
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Criar diretórios se não existirem
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)
