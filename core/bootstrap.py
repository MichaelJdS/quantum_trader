from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger

from core.logging_config import configure_logging
from core.settings import get_settings


async def bootstrap() -> None:
    """
    Inicializa todos os subsistemas na ordem correta:
      1. Configurações e logging.
      2. Diretórios necessários.
      3. Banco de dados (migrations + conexão).
      4. Cache in-memory.
      5. Registro de métricas Prometheus.
    """
    settings = get_settings()

    # 1. Logging deve ser o primeiro — tudo que vem depois usa o logger.
    configure_logging(settings)
    logger.info("Bootstrap iniciado", app=settings.app_name, env=settings.environment)

    # 2. Diretórios.
    _ensure_directories(settings)

    # 3. Banco de dados.
    from infra.db.database import init_db
    await init_db()
    logger.success("Banco de dados inicializado.")

    # 4. Cache.
    from infra.cache import FeatureCache
    FeatureCache.initialize(maxsize=2048)
    logger.success("Cache in-memory inicializado.")

    # 5. Groq Engine
    from ml.groq_engine import get_groq_engine
    if settings.groq_api_keys:
        get_groq_engine(api_keys=settings.groq_api_keys)
        logger.info("Groq Engine inicializado.", keys=len(settings.groq_api_keys))

    # 6. Prometheus (opcional — não bloqueia se não instalado).
    _start_metrics_server(settings)

    logger.success("Bootstrap concluído. Sistema pronto.")


def _ensure_directories(settings: "Settings") -> None:  # type: ignore[name-defined]
    """Cria diretórios obrigatórios caso não existam."""
    from core.settings import Settings
    s: Settings = settings
    dirs = [
        Path("models_store"),
        Path("logs"),
        Path("artifacts"),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    logger.debug("Diretórios verificados.", dirs=[str(d) for d in dirs])


def _start_metrics_server(settings: "Settings") -> None:  # type: ignore[name-defined]
    """Inicia servidor Prometheus em background (best-effort)."""
    try:
        from prometheus_client import start_http_server
        start_http_server(settings.prometheus_port)
        logger.info("Prometheus HTTP server iniciado.", port=settings.prometheus_port)
    except ImportError:
        logger.warning("prometheus_client não instalado — métricas desabilitadas.")
    except OSError as exc:
        logger.warning("Falha ao iniciar Prometheus server.", error=str(exc))