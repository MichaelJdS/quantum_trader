"""
cloud_api/metrics.py — Coletor de métricas para o endpoint /metrics
"""
from __future__ import annotations
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class MetricsCollector:
    _start_time:   float = field(default_factory=time.time, init=False)
    _trade_log:    deque = field(default_factory=lambda: deque(maxlen=200), init=False)
    _symbol_stats: dict  = field(default_factory=dict, init=False)
    _decisions:    deque = field(default_factory=lambda: deque(maxlen=50), init=False)

    def record_trade(self, event: dict) -> None:
        self._trade_log.append({**event, "recorded_at": time.time()})
        sym = event.get("symbol", "unknown")
        if sym not in self._symbol_stats:
            self._symbol_stats[sym] = {"wins": 0, "losses": 0, "pnl": 0.0}
        won = event.get("pnl", 0) > 0
        self._symbol_stats[sym]["wins" if won else "losses"] += 1
        self._symbol_stats[sym]["pnl"] += event.get("pnl", 0.0)

    def record_council_decision(self, decision: dict) -> None:
        self._decisions.append({**decision, "ts": datetime.now(tz=timezone.utc).isoformat()})

    def snapshot(self, engine: Any = None, circuit_breaker: Any = None, market_profiler: Any = None) -> dict:
        uptime_s  = int(time.time() - self._start_time)
        trades    = list(self._trade_log)
        total     = len(trades)
        wins      = sum(1 for t in trades if t.get("pnl", 0) > 0)
        total_pnl = sum(t.get("pnl", 0.0) for t in trades)
        by_symbol = [
            {"symbol": s, "wins": v["wins"], "losses": v["losses"],
             "win_rate": round(v["wins"] / max(v["wins"]+v["losses"], 1), 3),
             "pnl": round(v["pnl"], 4)}
            for s, v in self._symbol_stats.items()
        ]
        agent_weights = []
        if engine and engine.grand_oracle:
            try:
                agent_weights = engine.grand_oracle.get_council_health().get("agents", [])
            except Exception:
                pass
        return {
            "uptime_seconds": uptime_s,
            "uptime_human":   _fmt(uptime_s),
            "global": {
                "total_trades": total, "wins": wins,
                "losses": total - wins,
                "win_rate": round(wins / max(total, 1), 3),
                "total_pnl": round(total_pnl, 4),
            },
            "by_symbol":        by_symbol,
            "agent_weights":    agent_weights,
            "circuit_breaker":  circuit_breaker.status() if circuit_breaker else {},
            "market_profiles":  market_profiler.all_profiles() if market_profiler else {},
            "recent_decisions": list(self._decisions)[-10:],
            "recent_trades":    list(reversed(trades))[:20],
            "generated_at":     datetime.now(tz=timezone.utc).isoformat(),
        }


def _fmt(s: int) -> str:
    return f"{s//3600:02d}h {(s%3600)//60:02d}m {s%60:02d}s"
