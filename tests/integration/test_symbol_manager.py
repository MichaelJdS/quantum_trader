from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from infra.symbol_manager import SymbolManager


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.get_candles = AsyncMock(return_value=[
        {"open": "1000", "high": "1010", "low": "990", "close": "1005", "epoch": "1000000"},
        {"open": "1005", "high": "1015", "low": "995", "close": "1010", "epoch": "1000060"},
    ])
    client.subscribe_ticks = AsyncMock(return_value="sub_001")
    client.on = MagicMock()
    client.unsubscribe_all = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_initialize_loads_candles(mock_client):
    manager = SymbolManager(client=mock_client, symbols=["R_50"])
    
    await manager.initialize()

    assert "R_50" in manager.ready_symbols
    df = manager.get_candles_df("R_50")
    assert len(df) == 2
    assert "close" in df.columns


@pytest.mark.asyncio
async def test_get_recent_ticks_default(mock_client):
    manager = SymbolManager(client=mock_client, symbols=["R_75"])
    assert manager.get_recent_ticks("R_75") == []