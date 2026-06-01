from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.exceptions import DerivAuthorizationError, DerivContractError
from infra.deriv_client import CircuitBreaker, CircuitState, DerivClient, RateLimiter


# ── Circuit Breaker ───────────────────────────────────────────────────────────

def test_circuit_breaker_opens_after_threshold():
    cb = CircuitBreaker(threshold=3, reset_timeout=60.0)
    assert cb.allow_request() is True
    for _ in range(3):
        cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.allow_request() is False


def test_circuit_breaker_resets_after_timeout():
    import time
    cb = CircuitBreaker(threshold=1, reset_timeout=0.01)
    cb.record_failure()
    time.sleep(0.02)
    assert cb.allow_request() is True
    assert cb.state == CircuitState.HALF_OPEN


def test_circuit_breaker_closes_on_success():
    cb = CircuitBreaker(threshold=2)
    cb.record_failure()
    cb.record_failure()
    cb._opened_at = 0.0  # Força half-open.
    cb._state = CircuitState.HALF_OPEN
    cb.record_success()
    assert cb.state == CircuitState.CLOSED


# ── Rate Limiter ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rate_limiter_allows_under_limit():
    limiter = RateLimiter(rate=10, period=1.0)
    # Deve completar rapidamente para as primeiras 10 requisições.
    for _ in range(10):
        await limiter.acquire()


# ── Dry Run ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_buy_contract_dry_run():
    client = DerivClient(dry_run=True)
    result = await client.buy_contract(proposal_id="test123", price=1.0)
    assert "buy" in result
    assert result["buy"]["contract_id"].startswith("DRY_")


# ── Callbacks ─────────────────────────────────────────────────────────────────

def test_register_and_remove_callback():
    client = DerivClient(dry_run=True)

    async def my_callback(data): pass

    client.on("tick", my_callback)
    assert my_callback in client._callbacks.get("tick", [])

    client.off("tick", my_callback)
    assert my_callback not in client._callbacks.get("tick", [])