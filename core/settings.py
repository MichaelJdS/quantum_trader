from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.enums import Environment, StakeMode


class Settings(BaseSettings):
    """Configuração central da aplicação via variáveis de ambiente."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Aplicação ────────────────────────────────────────────────────────────
    app_env: Environment = Environment.DEVELOPMENT
    app_name: str = "quantum_trader"
    log_level: str = "INFO"

    # ── Deriv API ─────────────────────────────────────────────────────────────
    deriv_app_id: str
    deriv_api_token: str
    deriv_websocket_url: str = "wss://ws.binaryws.com/websockets/v3"

    # ── Símbolos ──────────────────────────────────────────────────────────────
    default_symbols: str = "R_50,R_75,R_100"
    default_granularity: int = Field(default=60, ge=1)
    account_currency: str = "USD"

    # ── Banco de dados ────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./quantum_trader.db"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    model_dir: str = "./models_store"

    # ── Risco ─────────────────────────────────────────────────────────────────
    max_daily_drawdown_pct: Annotated[float, Field(gt=0.0, le=1.0)] = 0.05
    max_consecutive_losses: Annotated[int, Field(ge=1)] = 5
    default_stop_win_pct: Annotated[float, Field(gt=0.0, le=1.0)] = 0.03
    default_stop_loss_pct: Annotated[float, Field(gt=0.0, le=1.0)] = 0.02
    default_stake: Annotated[float, Field(gt=0.0)] = 1.0
    stake_mode: StakeMode = StakeMode.FIXED
    kelly_fraction: Annotated[float, Field(gt=0.0, le=1.0)] = 0.25

    # ── Observabilidade ───────────────────────────────────────────────────────
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    prometheus_port: int = Field(default=9108, ge=1024, le=65535)

    # ── Propriedades derivadas ────────────────────────────────────────────────
    @property
    def is_production(self) -> bool:
        return self.app_env == Environment.PRODUCTION

    @property
    def symbols_list(self) -> list[str]:
        return [s.strip() for s in self.default_symbols.split(",") if s.strip()]

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"log_level deve ser um de: {allowed}")
        return v.upper()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retorna instância singleton das configurações (cacheada)."""
    return Settings()