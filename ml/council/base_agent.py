"""
ml/council/base_agent.py — Base v3.0: Cérebros Vivos

Novo: cada agente tem:
  - _memory: memória episódica de erros/acertos (aprende com trades passados)
  - _web_cache: cache de dados web (sentimento, notícias, eventos)
  - adapt(): auto-ajusta limiares internos baseado no histórico
  - get_introspection(): auto-diagnóstico para o dashboard
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from core.entities import SessionState, Signal


@dataclass
class AgentVote:
    """Voto de um agente especialista."""
    agent_name: str
    action:     str          # "BUY" | "SELL" | "NEUTRAL"
    score:      float        # 0.0–1.0
    veto:       bool  = False
    reasoning:  str   = ""
    meta:       dict  = field(default_factory=dict)  # NOVO: metadados extras

    def aligns_with(self, signal_direction: str) -> bool:
        return self.action == signal_direction or self.action == "NEUTRAL"

    def __repr__(self) -> str:
        veto_str = " [VETO]" if self.veto else ""
        return f"{self.agent_name}: {self.action} ({self.score:.2f}){veto_str} — {self.reasoning[:60]}"


@dataclass
class TradeMemory:
    """Memória de um trade passado para aprendizado do agente."""
    action:    str      # ação que o agente recomendou
    signal:    str      # ação que o sinal gerou
    won:       bool     # trade foi lucrativo?
    pnl:       float    # PnL do trade
    state_key: str      # hash do estado no momento
    timestamp: float    = field(default_factory=time.time)


class BaseAgent(ABC):
    """
    Interface v3.0 para todos os agentes do Oracle Council.
    
    Capacidades novas:
      - Memória episódica (últimos 500 trades)
      - Auto-adaptação de limiares (adapt())
      - Cache web compartilhado (15 min TTL)
      - Auto-diagnóstico (get_introspection())
      - Persistência em disco (save/load_state)
    """

    name:   str   = "BaseAgent"
    weight: float = 0.10

    # Configurações de aprendizado (sobrescreva nas subclasses)
    MEMORY_SIZE:    int   = 500
    ADAPT_EVERY:    int   = 25    # trades antes de re-adaptar
    MIN_SAMPLES:    int   = 15    # amostras mínimas para adaptar
    LEARN_RATE:     float = 0.05  # taxa de adaptação dos limiares

    # Cache web compartilhado entre TODOS os agentes (class-level)
    _shared_web_cache: dict[str, tuple[Any, float]] = {}
    _WEB_CACHE_TTL = 900  # 15 minutos

    def __init__(self) -> None:
        self._memory:       deque[TradeMemory] = deque(maxlen=self.MEMORY_SIZE)
        self._trade_count:  int   = 0
        self._win_count:    int   = 0
        self._adapt_count:  int   = 0
        self._last_vote:    AgentVote | None = None
        self._last_state:   str = ""
        self._thresholds:   dict[str, float] = self._default_thresholds()
        self._load_state()

    # ── API abstrata ──────────────────────────────────────────────────────────

    @abstractmethod
    def analyze(
        self,
        signal:   "Signal",
        df:       pd.DataFrame,
        session:  "SessionState",
        ticks:    list[dict] | None                  = None,
        peer_dfs: dict[str, pd.DataFrame] | None     = None,
    ) -> AgentVote:
        """Analisa e retorna voto."""
        ...

    def _default_thresholds(self) -> dict[str, float]:
        """Limiares padrão. Sobrescreva na subclasse para personalizar."""
        return {"confidence_floor": 0.50, "confidence_ceil": 0.92}

    # ── Aprendizado & Memória ─────────────────────────────────────────────────

    def record_outcome(self, action: str, signal: str, won: bool, pnl: float) -> None:
        """
        Registra o resultado de um trade na memória episódica.
        Chamado pelo GrandOracle após cada trade fechado.
        """
        mem = TradeMemory(
            action=action,
            signal=signal,
            won=won,
            pnl=pnl,
            state_key=self._last_state,
        )
        self._memory.append(mem)
        self._trade_count += 1
        if won:
            self._win_count += 1
        self._adapt_count += 1

        # Auto-adaptação periódica
        if self._adapt_count >= self.ADAPT_EVERY and len(self._memory) >= self.MIN_SAMPLES:
            self._adapt()
            self._adapt_count = 0
            self._save_state()

    def _adapt(self) -> None:
        """
        Auto-adapta limiares internos com base na memória episódica.
        Cada agente sobrescreve para lógica específica.
        """
        recent = list(self._memory)[-self.MIN_SAMPLES:]
        win_rate = sum(1 for m in recent if m.won) / len(recent)

        # Se win_rate < 45%: torna limiares mais conservadores
        # Se win_rate > 65%: torna limiares levemente mais permissivos
        if win_rate < 0.45:
            factor = 1.0 + self.LEARN_RATE
        elif win_rate > 0.65:
            factor = 1.0 - self.LEARN_RATE * 0.5
        else:
            factor = 1.0

        old_floor = self._thresholds.get("confidence_floor", 0.50)
        new_floor = max(0.40, min(0.80, old_floor * factor))
        self._thresholds["confidence_floor"] = new_floor

    def _state_key(self, df: pd.DataFrame) -> str:
        """Hash do estado de mercado atual (para busca na memória)."""
        try:
            last = df.iloc[-1]
            vals = [
                round(self._safe_float(last.get("rsi_14", 50.0)), 0),
                "U" if self._safe_float(last.get("ema_9", 1)) > self._safe_float(last.get("ema_21", 0)) else "D",
                round(self._safe_float(last.get("adx", 20.0)), -1),
            ]
            raw = "_".join(str(v) for v in vals)
            return hashlib.md5(raw.encode()).hexdigest()[:8]
        except Exception:
            return "unknown"

    def _recall_similar(self, state_key: str) -> tuple[float, int]:
        """
        Busca na memória trades com estado similar.
        Retorna (win_rate, n_amostras).
        """
        similar = [m for m in self._memory if m.state_key == state_key]
        if len(similar) < 3:
            return 0.5, 0
        wr = sum(1 for m in similar if m.won) / len(similar)
        return wr, len(similar)

    # ── Cache Web ─────────────────────────────────────────────────────────────

    @classmethod
    def _get_web_cache(cls, key: str) -> Any | None:
        entry = cls._shared_web_cache.get(key)
        if entry is None:
            return None
        data, ts = entry
        if time.time() - ts > cls._WEB_CACHE_TTL:
            del cls._shared_web_cache[key]
            return None
        return data

    @classmethod
    def _set_web_cache(cls, key: str, data: Any) -> None:
        cls._shared_web_cache[key] = (data, time.time())

    # ── Persistência ──────────────────────────────────────────────────────────

    def _state_path(self) -> str:
        base = os.path.join(os.path.dirname(__file__), "states")
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, f"{self.name.lower()}_state.json")

    def _save_state(self) -> None:
        try:
            data = {
                "thresholds":    self._thresholds,
                "trade_count":   self._trade_count,
                "win_count":     self._win_count,
                "memory": [
                    {
                        "action":    m.action,
                        "signal":    m.signal,
                        "won":       m.won,
                        "pnl":       m.pnl,
                        "state_key": m.state_key,
                        "timestamp": m.timestamp,
                    }
                    for m in self._memory
                ],
            }
            with open(self._state_path(), "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def _load_state(self) -> None:
        try:
            path = self._state_path()
            if not os.path.exists(path):
                return
            with open(path) as f:
                data = json.load(f)
            self._thresholds  = data.get("thresholds",  self._default_thresholds())
            self._trade_count = data.get("trade_count", 0)
            self._win_count   = data.get("win_count",   0)
            for m in data.get("memory", []):
                self._memory.append(TradeMemory(**m))
        except Exception:
            pass

    # ── Auto-diagnóstico ──────────────────────────────────────────────────────

    def get_introspection(self) -> dict:
        """Auto-diagnóstico completo para o dashboard."""
        recent_50 = list(self._memory)[-50:] if self._memory else []
        wr_50 = sum(1 for m in recent_50 if m.won) / max(len(recent_50), 1)
        avg_pnl = sum(m.pnl for m in recent_50) / max(len(recent_50), 1)
        
        last_action = self._last_vote.action if self._last_vote else "NEUTRAL"
        last_conf   = self._last_vote.score if self._last_vote else 0.0

        return {
            "agent":        self.name,
            "action":       last_action,
            "confidence":   last_conf,
            "trades_total": self._trade_count,
            "win_rate_50":  round(wr_50, 3),
            "avg_pnl_50":   round(avg_pnl, 4),
            "thresholds":   {k: round(v, 4) for k, v in self._thresholds.items()},
            "memory_size":  len(self._memory),
            "weight":       self.weight,
        }

    # ── Helper ────────────────────────────────────────────────────────────────

    def _safe_float(self, val, default: float = 0.0) -> float:
        try:
            v = float(val)
            return v if v == v else default
        except (TypeError, ValueError):
            return default