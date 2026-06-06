from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_LOG_DIR = BASE_DIR / "logs"


class Settings(BaseSettings):
    """
    Configurações centrais do Quantum Trader.

    Fontes:
      1. Variáveis de ambiente do sistema
      2. Arquivo .env na raiz do projeto
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Ambiente ──────────────────────────────────────────────────────────────
    app_name: str = Field(default="Quantum Trader", alias="APP_NAME")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=False, alias="DEBUG")

    # ── Deriv API ─────────────────────────────────────────────────────────────
    deriv_websocket_url: str = Field(
        default="wss://ws.derivws.com/websockets/v3",
        alias="DERIV_WEBSOCKET_URL",
    )
    deriv_app_id: int = Field(..., alias="DERIV_APP_ID")
    deriv_api_token: str = Field(..., alias="DERIV_API_TOKEN")

    # ── Operação ──────────────────────────────────────────────────────────────
    dry_run_default: bool = Field(default=True, alias="DRY_RUN_DEFAULT")
    default_symbols: str = Field(default="R_50,R_75", alias="DEFAULT_SYMBOLS")
    default_granularity: int = Field(default=60, alias="DEFAULT_GRANULARITY")

    # ── Rede / retries ────────────────────────────────────────────────────────
    max_retries: int = Field(default=10, alias="MAX_RETRIES")
    request_timeout_seconds: float = Field(default=15.0, alias="REQUEST_TIMEOUT_SECONDS")
    reconnect_delay_seconds: float = Field(default=2.0, alias="RECONNECT_DELAY_SECONDS")

    # ── Banco / paths ─────────────────────────────────────────────────────────
    database_url: str = Field(default="sqlite+aiosqlite:///data/quantum_trader.db", alias="DATABASE_URL")
    data_dir: str = Field(default=str(DEFAULT_DATA_DIR), alias="DATA_DIR")
    log_dir: str = Field(default=str(DEFAULT_LOG_DIR), alias="LOG_DIR")

    # ── Gemini / Council ──────────────────────────────────────────────────────
    gemini_enabled: bool = Field(default=False, alias="GEMINI_ENABLED")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-pro", alias="GEMINI_MODEL")
    gemini_consult_interval_seconds: int = Field(default=60, alias="GEMINI_CONSULT_INTERVAL_SECONDS")

    # ── Cloud / API ───────────────────────────────────────────────────────────
    api_host: str = Field(default="127.0.0.1", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        allowed = {"development", "test", "staging", "production"}
        normalized = value.strip().lower()
        if normalized not in allowed:
            raise ValueError(f"ENVIRONMENT inválido: {value!r}. Use um de {sorted(allowed)}")
        return normalized

    @field_validator("deriv_websocket_url")
    @classmethod
    def validate_ws_url(cls, value: str) -> str:
        v = value.strip()
        if not v.startswith(("ws://", "wss://")):
            raise ValueError("DERIV_WEBSOCKET_URL deve começar com ws:// ou wss://")
        return v.rstrip("/")

    @field_validator("deriv_api_token")
    @classmethod
    def validate_token(cls, value: str) -> str:
        v = value.strip()
        if not v:
            raise ValueError("DERIV_API_TOKEN não pode ser vazio")
        return v

    @field_validator("deriv_app_id")
    @classmethod
    def validate_app_id(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("DERIV_APP_ID deve ser > 0")
        return value

    @field_validator("default_granularity")
    @classmethod
    def validate_granularity(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("DEFAULT_GRANULARITY deve ser > 0")
        return value

    @field_validator("max_retries")
    @classmethod
    def validate_max_retries(cls, value: int) -> int:
        if value < 1:
            raise ValueError("MAX_RETRIES deve ser >= 1")
        return value

    @field_validator("request_timeout_seconds", "reconnect_delay_seconds")
    @classmethod
    def validate_positive_float(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Valor deve ser > 0")
        return value

    @field_validator("gemini_api_key")
    @classmethod
    def normalize_optional_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def default_symbols_list(self) -> list[str]:
        return [
            symbol.strip()
            for symbol in self.default_symbols.split(",")
            if symbol.strip()
        ]

    @property
    def cors_origins_list(self) -> list[str]:
        raw = self.cors_origins.strip()
        if raw == "*":
            return ["*"]
        return [item.strip() for item in raw.split(",") if item.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    def ensure_directories(self) -> None:
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    def safe_dict(self) -> dict:
        data = self.model_dump()
        if "deriv_api_token" in data:
            data["deriv_api_token"] = "***"
        if "gemini_api_key" in data and data["gemini_api_key"]:
            data["gemini_api_key"] = "***"
        return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Carrega e cacheia settings.
    Falha cedo com mensagem clara se houver configuração inválida.
    """
    try:
        settings = Settings()
        settings.ensure_directories()
        return settings
    except ValidationError as exc:
        raise RuntimeError(
            "Falha ao carregar configurações. Verifique .env/variáveis de ambiente."
        ) from exc