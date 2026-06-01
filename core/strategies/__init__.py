"""Estratégias de trading disponíveis no Quantum Trader."""

from core.strategies.bollinger_reversion import BollingerReversionStrategy
from core.strategies.breakout import BreakoutStrategy
from core.strategies.ema_rsi import EmaRsiStrategy

__all__ = [
    "EmaRsiStrategy",
    "BollingerReversionStrategy",
    "BreakoutStrategy",
]