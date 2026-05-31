import datetime
from typing import Dict, Any
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from src.config import settings

Base = declarative_base()

class TradeLog(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    symbol = Column(String(20), nullable=False)
    direction = Column(String(4), nullable=False)
    stake = Column(Float, nullable=False)
    entry_price = Column(Float)
    exit_price = Column(Float)
    pnl = Column(Float, default=0.0)
    agent_votes = Column(JSON)
    consensus_score = Column(Float)
    status = Column(String(10), default="open")

class AgentMetric(Base):
    __tablename__ = "agent_metrics"
    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_name = Column(String(50), nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    confidence = Column(Float)
    accuracy = Column(Float)
    trades = Column(Integer, default=0)
    regime = Column(String(20))

engine = create_async_engine(settings.DB_URL, echo=False, pool_size=10, max_overflow=5)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)