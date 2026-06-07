from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from core.settings import Settings


def configure_logging(settings: "Settings") -> None:
    """
    Configura o Loguru com base nas settings da aplicação.

    Saídas:
      - stderr (colorido, humano) em desenvolvimento.
      - arquivo rotativo JSON em staging/produção.
    """
    logger.remove()  # Remove o handler padrão do Loguru.

    # ── Handler de console ────────────────────────────────────────────────────
    fmt_dev = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    )
    fmt_prod = "{time:YYYY-MM-DDTHH:mm:ss.SSSZ} | {level} | {name}:{function}:{line} | {message}"

    logger.add(
        sys.stderr,
        format=fmt_dev if not settings.is_production else fmt_prod,
        level=settings.log_level,
        colorize=not settings.is_production,
        backtrace=True,
        diagnose=not settings.is_production,
    )

    # ── Handler de arquivo (rotativo) ─────────────────────────────────────────
    if settings.is_production or settings.environment == "staging":
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        logger.add(
            log_dir / f"{settings.app_name}.jsonl",
            format="{time:YYYY-MM-DDTHH:mm:ss.SSSZ} | {level} | {name}:{function}:{line} | {message}",
            level=settings.log_level,
            rotation="100 MB",
            retention="30 days",
            compression="gz",
            serialize=True,  # Grava como JSON Lines.
            enqueue=True,    # Thread-safe / async-safe.
            backtrace=True,
            diagnose=False,  # Sem dados sensíveis em prod.
        )

    logger.info(
        "Logging configurado",
        env=settings.environment,
        level=settings.log_level,
    )