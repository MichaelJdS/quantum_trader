import os
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    DERIV_APP_ID: int = 1089
    DERIV_TOKEN: str = ""
    DB_URL: str = "sqlite+aiosqlite:///quantum_trader.db"
    REDIS_URL: str = "redis://localhost:6379/0"
    MAX_DRAWDOWN_PCT: float = 5.0
    MAX_POSITION_PCT: float = 2.0
    CONSENSUS_THRESHOLD: float = 0.65
    QUANTUM_WEIGHT_DECAY: float = 0.98
    SYMBOL: str = "R_100"
    
    # Configuração de tempo atualizada para 1 Minuto
    DURATION: int = 1
    DURATION_UNIT: str = "m"  # "m" para minutos, "t" para ticks, "s" para segundos
    
    STAKE: float = 10.0
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True

    @field_validator("DERIV_TOKEN")
    @classmethod
    def validate_token(cls, v):
        if not v or v == "SEU_TOKEN_DEMO_AQUI":
            raise ValueError("Configure DERIV_TOKEN no .env antes de executar")
        return v


settings = Settings()