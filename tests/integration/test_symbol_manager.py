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
    with patch("infra.symbol_manager.get_session"):
        manager = SymbolManager(client=mock_client, symbols=["R_50"])
        # Simula get_session como no-op.
        with patch("infra.symbol_manager.CandleRepository") as mock_repo_cls:
            mock_repo = AsyncMock()
            mock_repo_cls.return_value = mock_repo
            mock_repo.bulk_upsert = AsyncMock()

            with patch("infra.symbol_manager.get_session") as mock_gs:
                mock_ctx = AsyncMock()
                mock_ctx.__aenter__ = AsyncMock(return_value=mock_repo)
                mock_ctx.__aexit__ = AsyncMock(return_value=False)
                mock_gs.return_value = mock_ctx

                await manager.initialize()

    assert manager.is_symbol_ready("R_50") is True
    df = manager.get_candles_df("R_50")
    assert len(df) == 2
    assert "close" in df.columns


@pytest.mark.asyncio
async def test_get_last_price_default(mock_client):
    manager = SymbolManager(client=mock_client, symbols=["R_75"])
    assert manager.get_last_price("R_75") == 0.0


@pytest.mark.asyncio
async def test_shutdown(mock_client):
    manager = SymbolManager(client=mock_client, symbols=["R_50"])
    await manager.shutdown()
    mock_client.unsubscribe_all.assert_called_once()