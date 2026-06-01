"""
Configuracao do banco de dados SQLAlchemy async para SQLite.

SQLite com aiosqlite usa NullPool por padrao â€” nao aceita
pool_size nem max_overflow. Esses parametros sao validos
apenas para PostgreSQL/MySQL.
"""
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
from sqlalchemy.pool import NullPool, StaticPool

from core.settings import get_settings

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _build_engine(database_url: str) -> AsyncEngine:
    """
    Cria o engine async correto de acordo com o banco.

    SQLite (aiosqlite): NullPool â€” sem pool_size / max_overflow.
    SQLite :memory:   : StaticPool â€” mantÃ©m conexao unica persistente.
    PostgreSQL/MySQL  : pool_size + max_overflow normais.
    """
    is_sqlite = "sqlite" in database_url
    is_memory = ":memory:" in database_url

    if is_memory:
        # Banco em memoria: StaticPool para manter estado entre requests.
        return create_async_engine(
            database_url,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
            echo=False,
        )

    if is_sqlite:
        # SQLite em arquivo: NullPool (sem pool_size).
        return create_async_engine(
            database_url,
            poolclass=NullPool,
            connect_args={"check_same_thread": False},
            echo=False,
        )

    # PostgreSQL / MySQL: pool completo.
    return create_async_engine(
        database_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        echo=False,
    )


async def init_db() -> None:
    """Inicializa o engine e cria todas as tabelas."""
    global _engine, _session_factory

    settings = get_settings()
    database_url = settings.database_url

    logger.info("Inicializando banco de dados.", url=database_url)

    _engine = _build_engine(database_url)
    _session_factory = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Cria tabelas se nao existirem.
    from infra.db.models_db import Base
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.success("Banco de dados inicializado com sucesso.")


async def close_db() -> None:
    """Fecha o engine ao encerrar a aplicacao."""
    global _engine
    if _engine:
        await _engine.dispose()
        logger.info("Conexao com banco de dados encerrada.")


def get_engine() -> AsyncEngine:
    if _engine is None:
        raise RuntimeError("Banco nao inicializado. Chame init_db() primeiro.")
    return _engine


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Context manager para session async â€” commit/rollback automatico."""
    if _session_factory is None:
        raise RuntimeError("Banco nao inicializado. Chame init_db() primeiro.")

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
