"""
core/circuit_breaker.py — Circuit Breaker Global do Quantum Trader
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class CircuitBreaker:
    max_consecutive_losses: int   = 5
    max_drawdown_pct:       float = 0.15   # 15%
    max_hourly_losses:      int   = 8
    cooldown_seconds:       int   = 1800   # 30 min

    _tripped:       bool  = field(default=False, init=False)
    _trip_reason:   str   = field(default="",    init=False)
    _trip_time:     float = field(default=0.0,   init=False)
    _hourly_losses: list  = field(default_factory=list, init=False)

    @property
    def is_open(self) -> bool:
        if not self._tripped:
            return False
        if time.time() - self._trip_time >= self.cooldown_seconds:
            self._reset()
            return False
        return True

    @property
    def trip_reason(self) -> str:
        return self._trip_reason

    @property
    def seconds_until_reset(self) -> int:
        if not self._tripped:
            return 0
        return max(0, int(self.cooldown_seconds - (time.time() - self._trip_time)))

    def check(self, consecutive_losses: int, current_balance: float, initial_balance: float) -> bool:
        if self._tripped:
            return False
        if consecutive_losses >= self.max_consecutive_losses:
            self._trip(f"{consecutive_losses} perdas consecutivas")
            return True
        if initial_balance > 0:
            drawdown = (initial_balance - current_balance) / initial_balance
            if drawdown >= self.max_drawdown_pct:
                self._trip(f"Drawdown de {drawdown*100:.1f}% atingido")
                return True
        now = time.time()
        self._hourly_losses = [t for t in self._hourly_losses if now - t < 3600]
        if len(self._hourly_losses) >= self.max_hourly_losses:
            self._trip(f"{len(self._hourly_losses)} perdas na última hora")
            return True
        return False

    def register_loss(self) -> None:
        self._hourly_losses.append(time.time())

    def force_reset(self) -> None:
        self._reset()

    def _trip(self, reason: str) -> None:
        self._tripped     = True
        self._trip_reason = reason
        self._trip_time   = time.time()
        logger.critical("🔴 Circuit Breaker ATIVADO.", reason=reason, cooldown_min=self.cooldown_seconds//60)

    def _reset(self) -> None:
        self._tripped     = False
        self._trip_reason = ""
        self._trip_time   = 0.0
        self._hourly_losses.clear()
        logger.info("🟢 Circuit Breaker RESETADO.")

    def status(self) -> dict:
        return {
            "tripped":             self._tripped,
            "reason":              self._trip_reason,
            "seconds_until_reset": self.seconds_until_reset,
            "hourly_losses":       len(self._hourly_losses),
            "config": {
                "max_consecutive_losses": self.max_consecutive_losses,
                "max_drawdown_pct":       self.max_drawdown_pct,
                "max_hourly_losses":      self.max_hourly_losses,
                "cooldown_minutes":       self.cooldown_seconds // 60,
            },
        }
