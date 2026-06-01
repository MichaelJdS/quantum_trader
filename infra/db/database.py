from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from loguru import logger
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from infra.db.models_db import Base

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _get_engine() -> AsyncEngine:
    """Retorna a engine singleton, criando se necessário."""
    global _engine
    if _engine is None:
        raise RuntimeError("Banco não inicializado. Chame `init_db()` antes.")
    return _engine


def _get_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        raise RuntimeError("Banco não inicializado. Chame `init_db()` antes.")
    return _session_factory


async def init_db() -> None:
    """
    Inicializa engine e session factory, e garante que todas as tabelas existam.
    Deve ser chamado UMA VEZ no bootstrap da aplicação.
    """
    global _engine, _session_factory

    from core.settings import get_settings
    settings = get_settings()

    _engine = create_async_engine(
        settings.database_url,
        echo=not settings.is_production,
        pool_pre_ping=True,
        pool_size=5 if "postgresql" in settings.database_url else 1,
        max_overflow=10 if "postgresql" in settings.database_url else 0,
        connect_args={"check_same_thread": False}
        if "sqlite" in settings.database_url
        else {},
    )

    _session_factory = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.success(
        "Database inicializado.",
        url=settings.database_url.split("///")[0] + "///***",
    )


async def close_db() -> None:
    """Fecha a engine — deve ser chamado no shutdown da aplicação."""
    global _engine
    if _engine:
        await _engine.dispose()
        logger.info("Database connection pool fechado.")
        _engine = None


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager para obter sessão assíncrona com transação automática.

    Uso:
        async with get_session() as session:
            result = await session.execute(...)

    Faz commit automático se não houver exceções; rollback em caso de erro.
    """
    factory = _get_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise