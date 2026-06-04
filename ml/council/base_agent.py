"""
ml/council/base_agent.py — Interface Abstrata do Conselho Oracle

Cada agente especialista retorna um AgentVote com:
- action:    "BUY" | "SELL" | "NEUTRAL"
- score:     0.0 (sem confiança) → 1.0 (máxima confiança)
- veto:      True → bloqueia o trade independente dos demais votos
- reasoning: justificativa curta para logs e UI
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from core.entities import SessionState, Signal


@dataclass
class AgentVote:
    """Voto de um agente especialista."""
    agent_name: str
    action: str          # "BUY" | "SELL" | "NEUTRAL"
    score: float         # 0.0–1.0
    veto: bool = False   # veto absoluto → bloqueia o trade
    reasoning: str = ""

    def aligns_with(self, signal_direction: str) -> bool:
        """Verifica se o voto está alinhado com o sinal da estratégia."""
        return self.action == signal_direction or self.action == "NEUTRAL"

    def __repr__(self) -> str:
        veto_str = " [VETO]" if self.veto else ""
        return f"{self.agent_name}: {self.action} ({self.score:.2f}){veto_str} — {self.reasoning[:60]}"


class BaseAgent(ABC):
    """Interface abstrata para todos os 8 agentes do Oracle Council."""

    name: str = "BaseAgent"
    weight: float = 0.10   # Peso no Grand Oracle (soma = 1.0)

    @abstractmethod
    def analyze(
        self,
        signal: "Signal",
        df: pd.DataFrame,
        session: "SessionState",
        ticks: list[dict] | None = None,
        peer_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> AgentVote:
        """
        Analisa o mercado e retorna um voto.

        Args:
            signal:    Sinal da estratégia principal (CALL/PUT, confiança base)
            df:        DataFrame de features do símbolo (últimos N candles)
            session:   Estado atual da sessão (balance, trades, losses etc.)
            ticks:     Lista de ticks recentes (opcional, para ARES)
            peer_dfs:  DataFrames de outros símbolos (opcional, para LUMEN)

        Returns:
            AgentVote com action, score, veto e reasoning
        """
        ...

    def _safe_float(self, val, default: float = 0.0) -> float:
        """Helper: converte valor para float com fallback seguro."""
        try:
            v = float(val)
            return v if not (v != v) else default  # NaN check
        except (TypeError, ValueError):
            return default
