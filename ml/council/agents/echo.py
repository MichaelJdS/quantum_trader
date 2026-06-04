"""
ml/council/agents/echo.py — ECHO: Aprendizado por Reforço (Q-Table)

Implementa um agente Q-Learning tabular simples que aprende com cada trade.
Estado = (RSI_bucket, EMA_direction, hora_do_dia)
Ação   = BUY | SELL | NEUTRAL
Começa neutro e melhora progressivamente com o histórico da sessão.

Fase 2: será substituído por PPO/SAC com Stable-Baselines3.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import TYPE_CHECKING

import pandas as pd

from ml.council.base_agent import AgentVote, BaseAgent

_Q_TABLE_PATH = os.path.join(os.path.dirname(__file__), "echo_qtable.json")


class EchoAgent(BaseAgent):
    """Especialista em Aprendizado por Reforço — ECHO."""

    name = "ECHO"
    weight = 0.08

    LEARNING_RATE = 0.15
    DISCOUNT      = 0.85
    ACTIONS       = ["BUY", "SELL", "NEUTRAL"]

    def __init__(self) -> None:
        self._q: dict = defaultdict(lambda: {"BUY": 0.0, "SELL": 0.0, "NEUTRAL": 0.0})
        self._last_state: str | None = None
        self._last_action: str | None = None
        self._total_updates: int = 0
        self._load_qtable()

    def analyze(self, signal, df, session, ticks=None, peer_dfs=None) -> AgentVote:
        state = self._get_state(df, session)

        # Q-values para este estado
        q_vals = self._q[state]
        best_action = max(q_vals, key=lambda a: q_vals[a])
        best_q = q_vals[best_action]

        # Score baseado na diferença entre o melhor e os demais Q-values
        q_max = max(q_vals.values())
        q_min = min(q_vals.values())
        q_range = q_max - q_min if q_max != q_min else 1.0

        if self._total_updates < 10:
            # Ainda aprendendo — neutro com baixa confiança
            return AgentVote(
                self.name, "NEUTRAL", 0.5,
                reasoning=f"Aprendendo... ({self._total_updates} updates)"
            )

        confidence = 0.5 + min((q_max - q_min) / max(abs(q_max) + 0.01, 1.0), 0.40)
        confidence = max(0.3, min(confidence, 0.85))

        self._last_state  = state
        self._last_action = best_action

        return AgentVote(
            self.name, best_action, confidence,
            reasoning=f"Q-table: estado={state} BUY={q_vals['BUY']:.2f} SELL={q_vals['SELL']:.2f} → {best_action}"
        )

    def update_from_trade(self, action: str, reward: float) -> None:
        """
        Atualiza os Q-values com o resultado do trade.
        Chamado pelo ExecutionEngine após _await_result().
        reward > 0 = trade lucrativo, reward < 0 = perda.
        """
        if self._last_state is None or self._last_action is None:
            return

        state = self._last_state
        q_old = self._q[state][action]
        q_next_max = max(self._q[state].values())  # próximo estado = mesmo (simplificação)

        # Bellman update
        q_new = q_old + self.LEARNING_RATE * (
            reward + self.DISCOUNT * q_next_max - q_old
        )
        self._q[state][action] = round(q_new, 4)
        self._total_updates += 1

        # Salva a tabela periodicamente
        if self._total_updates % 10 == 0:
            self._save_qtable()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_state(self, df: pd.DataFrame, session) -> str:
        """Discretiza o estado em uma string para a Q-table."""
        if len(df) < 2:
            return "neutral_neutral_12"

        last = df.iloc[-1]
        rsi = self._safe_float(last.get("rsi_14", 50.0), 50.0)
        e9  = self._safe_float(last.get("ema_9",  last.get("close", 0)), 0)
        e21 = self._safe_float(last.get("ema_21", last.get("close", 0)), 0)

        # Bucket RSI: low / mid_low / mid / mid_high / high
        if rsi < 35:       rsi_bucket = "rsi_low"
        elif rsi < 45:     rsi_bucket = "rsi_midlow"
        elif rsi < 55:     rsi_bucket = "rsi_mid"
        elif rsi < 65:     rsi_bucket = "rsi_midhigh"
        else:              rsi_bucket = "rsi_high"

        # EMA direction
        ema_dir = "bull" if e9 > e21 else "bear"

        # Hora do dia (simplificado em blocos de 4h)
        hour_block = (int(time.localtime().tm_hour) // 4) * 4

        return f"{rsi_bucket}_{ema_dir}_{hour_block:02d}"

    def _load_qtable(self) -> None:
        """Carrega Q-table salva em disco (persistência entre sessões)."""
        try:
            if os.path.exists(_Q_TABLE_PATH):
                with open(_Q_TABLE_PATH, "r") as f:
                    data = json.load(f)
                for state, vals in data.get("q", {}).items():
                    self._q[state] = vals
                self._total_updates = data.get("total_updates", 0)
        except Exception:
            pass  # Q-table corrompida → começa do zero

    def _save_qtable(self) -> None:
        """Salva Q-table em disco para persistência entre sessões."""
        try:
            with open(_Q_TABLE_PATH, "w") as f:
                json.dump({
                    "q": dict(self._q),
                    "total_updates": self._total_updates,
                }, f, indent=2)
        except Exception:
            pass
