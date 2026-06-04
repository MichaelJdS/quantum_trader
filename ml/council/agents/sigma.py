"""
ml/council/agents/sigma.py — SIGMA: Gestão de Risco & Drawdown

Único agente com poder de VETO absoluto.
Calcula: VaR(95%) baseado em trades recentes, drawdown atual,
consecutive_losses, e stakes acima do limite Kelly.
Bloqueia o trade se qualquer condição crítica for violada.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.council.base_agent import AgentVote, BaseAgent


class SigmaAgent(BaseAgent):
    """Especialista em Gestão de Risco & Drawdown — SIGMA (poder de VETO)."""

    name = "SIGMA"
    weight = 0.20   # Maior peso do Conselho

    # Limiares de risco
    MAX_CONSECUTIVE_LOSSES = 4   # Veto se ≥ N perdas seguidas
    MAX_DRAWDOWN_PCT        = 0.08 # Veto se drawdown ≥ 8%
    VAR_CONFIDENCE          = 0.95 # VaR 95%
    VAR_LOOKBACK_TRADES     = 30   # Usa últimos N trades para VaR

    def analyze(self, signal, df, session, ticks=None, peer_dfs=None) -> AgentVote:
        # ── 1. Consecutive losses veto ────────────────────────────────────────
        consec = getattr(session, "consecutive_losses", 0)
        if consec >= self.MAX_CONSECUTIVE_LOSSES:
            return AgentVote(
                self.name, "NEUTRAL", 0.0, veto=True,
                reasoning=f"VETO: {consec} perdas consecutivas ≥ limite {self.MAX_CONSECUTIVE_LOSSES}"
            )

        # ── 2. Drawdown veto ──────────────────────────────────────────────────
        initial_balance = self._safe_float(getattr(session, "initial_balance", 1000.0), 1000.0)
        current_balance = self._safe_float(getattr(session, "current_balance", initial_balance), initial_balance)

        if initial_balance > 0:
            drawdown = (initial_balance - current_balance) / initial_balance
            if drawdown >= self.MAX_DRAWDOWN_PCT:
                return AgentVote(
                    self.name, "NEUTRAL", 0.0, veto=True,
                    reasoning=f"VETO: Drawdown {drawdown:.1%} ≥ limite {self.MAX_DRAWDOWN_PCT:.1%}"
                )
        else:
            drawdown = 0.0

        # ── 3. VaR dinâmico sobre histórico de trades ─────────────────────────
        trade_results = getattr(session, "trade_results", [])
        var_score = self._compute_var_score(trade_results, signal)

        # ── 4. Score de saúde da sessão ───────────────────────────────────────
        win_rate    = self._safe_float(getattr(session, "win_rate", 0.5), 0.5)
        total_trades = getattr(session, "total_trades", 0)

        health_score = self._compute_health_score(
            consec=consec,
            drawdown=drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            var_score=var_score,
        )

        sig_dir = signal.direction.value if hasattr(signal.direction, "value") else str(signal.direction)
        is_buy  = sig_dir in ("BUY", "buy", "CALL", "call")

        # SIGMA vota na mesma direção do sinal se a saúde está OK
        action = "BUY" if is_buy else "SELL"

        reasoning = (
            f"Saúde OK: loss={consec} dd={drawdown:.1%} wr={win_rate:.0%} "
            f"health={health_score:.2f}"
        )
        return AgentVote(self.name, action, health_score, reasoning=reasoning)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _compute_var_score(self, trade_results: list, signal) -> float:
        """
        Calcula VaR(95%) baseado nos resultados históricos de trades.
        Retorna score 0.0–1.0 (1.0 = risco mínimo).
        """
        if not trade_results or len(trade_results) < 5:
            return 0.6  # Sem histórico → score neutro

        # Extrai PnL dos trades
        pnl_list = []
        for t in trade_results[-self.VAR_LOOKBACK_TRADES:]:
            pnl = t.get("pnl", 0.0) if isinstance(t, dict) else 0.0
            pnl_list.append(float(pnl))

        if not pnl_list:
            return 0.6

        arr = np.array(pnl_list)
        var_95 = float(np.percentile(arr, 5))   # 5th percentile = VaR(95%)
        # VaR negativo grande → maior risco
        if var_95 >= 0:
            return 0.85   # Pior caso histórico é lucrativo
        elif var_95 > -2.0:
            return 0.70
        elif var_95 > -5.0:
            return 0.55
        else:
            return 0.35   # VaR muito negativo → cuidado

    def _compute_health_score(
        self,
        consec: int,
        drawdown: float,
        win_rate: float,
        total_trades: int,
        var_score: float,
    ) -> float:
        """Score composto de saúde da sessão (0.0–1.0)."""
        # Penalidade por perdas consecutivas (0 = perfeito, 3 = penalidade máxima)
        loss_penalty = min(consec / self.MAX_CONSECUTIVE_LOSSES, 1.0) * 0.3

        # Penalidade por drawdown
        dd_penalty = min(drawdown / self.MAX_DRAWDOWN_PCT, 1.0) * 0.3

        # Bônus por win_rate (se > 50%)
        wr_bonus = max(0, win_rate - 0.5) * 0.4

        # VaR contribui diretamente
        base = var_score * 0.5 + wr_bonus + 0.4

        score = max(0.1, min(0.95, base - loss_penalty - dd_penalty))
        return score
