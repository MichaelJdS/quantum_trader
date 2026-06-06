"""
ml/council/agents/sigma.py — SIGMA v3.0: Guardião de Risco Adaptativo

Melhorias v3.0:
  - CVaR (Conditional VaR) além do VaR simples — mede o pior 5%
  - Kelly Criterion dinâmico: calcula o stake ideal e veta se exceder
  - Auto-adaptação: reduz MAX_DRAWDOWN e MAX_CONSEC conforme histórico
  - Memória episódica: lembra de contextos onde o risco foi subestimado
  - Volatility Regime Adjustment: em mercados voláteis, limiares mais rígidos
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from ml.council.base_agent import AgentVote, BaseAgent


class SigmaAgent(BaseAgent):
    """SIGMA v3.0 — Guardião de Risco Adaptativo com VETO."""

    name   = "SIGMA"
    weight = 0.20
    ADAPT_EVERY = 20

    def _default_thresholds(self) -> dict[str, float]:
        return {
            "max_consecutive_losses": 4.0,
            "max_drawdown_pct":       0.08,
            "var_lookback":           30.0,
            "kelly_safety_factor":    0.25,   # usa 25% do Kelly ótimo
            "cvar_block_threshold":   -0.05,  # CVaR pior que -5% → veto
            "confidence_floor":       0.50,
        }

    def analyze(self, signal, df, session, ticks=None, peer_dfs=None) -> AgentVote:
        t = self._thresholds
        self._last_state = self._state_key(df)

        # ── 1. Consecutive losses VETO ────────────────────────────────────────
        consec = getattr(session, "consecutive_losses", 0)
        max_consec = int(t["max_consecutive_losses"])
        if consec >= max_consec:
            return AgentVote(
                self.name, "NEUTRAL", 0.0, veto=True,
                reasoning=f"VETO: {consec} perdas consecutivas ≥ limite {max_consec}",
                meta={"trigger": "consecutive_losses", "value": consec},
            )

        # ── 2. Drawdown VETO ──────────────────────────────────────────────────
        initial  = self._safe_float(getattr(session, "initial_balance", 1000.0), 1000.0)
        current  = self._safe_float(getattr(session, "current_balance", initial), initial)
        drawdown = (initial - current) / initial if initial > 0 else 0.0
        max_dd   = t["max_drawdown_pct"]

        if drawdown >= max_dd:
            return AgentVote(
                self.name, "NEUTRAL", 0.0, veto=True,
                reasoning=f"VETO: Drawdown {drawdown:.1%} ≥ {max_dd:.1%}",
                meta={"trigger": "drawdown", "value": round(drawdown, 4)},
            )

        # ── 3. CVaR check ─────────────────────────────────────────────────────
        trade_results = getattr(session, "trade_results", [])
        cvar, var_score = self._compute_cvar(trade_results, t)

        cvar_limit = t["cvar_block_threshold"]
        if cvar < cvar_limit and len(trade_results) >= 10:
            return AgentVote(
                self.name, "NEUTRAL", 0.0, veto=True,
                reasoning=f"VETO: CVaR {cvar:.3f} < limite {cvar_limit:.3f} — perda sistêmica",
                meta={"trigger": "cvar", "cvar": round(cvar, 4)},
            )

        # ── 4. Kelly Criterion dinâmico ───────────────────────────────────────
        kelly_score = self._check_kelly(signal, session, t)

        # ── 5. Volatility regime adjustment ──────────────────────────────────
        vol_penalty = self._vol_regime_penalty(df)

        # ── 6. Memória episódica: recall de contextos similares ───────────────
        similar_wr, n_similar = self._recall_similar(self._last_state)
        memory_boost = 0.0
        if n_similar >= 5:
            memory_boost = (similar_wr - 0.5) * 0.15

        # ── 7. Score composto de saúde ────────────────────────────────────────
        win_rate     = self._safe_float(getattr(session, "win_rate", 0.5), 0.5)
        total_trades = getattr(session, "total_trades", 0)

        health = self._compute_health(
            consec=consec,
            drawdown=drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            var_score=var_score,
            kelly_score=kelly_score,
            vol_penalty=vol_penalty,
            memory_boost=memory_boost,
        )

        sig_dir = signal.direction.value if hasattr(signal.direction, "value") else str(signal.direction)
        is_buy  = sig_dir in ("BUY", "buy", "CALL", "call")
        action  = "BUY" if is_buy else "SELL"

        floor = t["confidence_floor"]
        health = max(floor, min(0.95, health))

        self._last_vote = AgentVote(
            self.name, action, health,
            reasoning=(
                f"Saúde={health:.2f} dd={drawdown:.1%} consec={consec} "
                f"wr={win_rate:.0%} cvar={cvar:.3f} vol_pen={vol_penalty:.2f} "
                f"mem={similar_wr:.2f}(n={n_similar})"
            ),
            meta={
                "drawdown": round(drawdown, 4),
                "cvar":     round(cvar, 4),
                "kelly_ok": kelly_score > 0.5,
            },
        )
        return self._last_vote

    # ── Adaptação ─────────────────────────────────────────────────────────────

    def _adapt(self) -> None:
        """Ajusta limiares com base em perdas recentes."""
        super()._adapt()
        recent = list(self._memory)[-self.MIN_SAMPLES:]
        if not recent:
            return

        avg_pnl   = sum(m.pnl for m in recent) / len(recent)
        consec_max = 0
        cur_consec = 0
        for m in recent:
            if not m.won:
                cur_consec += 1
                consec_max = max(consec_max, cur_consec)
            else:
                cur_consec = 0

        # Se houve sequências longas de perdas, reduz o MAX_CONSEC
        if consec_max >= self._thresholds["max_consecutive_losses"]:
            self._thresholds["max_consecutive_losses"] = max(
                2.0,
                self._thresholds["max_consecutive_losses"] - 0.5
            )
            logger.info(
                "SIGMA auto-adaptou: max_consec reduzido.",
                new_val=self._thresholds["max_consecutive_losses"],
            )

        # Se avg_pnl muito negativo, aperta o drawdown
        if avg_pnl < -0.02:
            self._thresholds["max_drawdown_pct"] = max(
                0.04,
                self._thresholds["max_drawdown_pct"] - 0.005
            )
            logger.info(
                "SIGMA auto-adaptou: max_drawdown reduzido.",
                new_val=self._thresholds["max_drawdown_pct"],
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _compute_cvar(self, trade_results: list, t: dict) -> tuple[float, float]:
        """
        Calcula CVaR (Expected Shortfall) — média do pior 5% dos trades.
        Retorna (cvar, var_score).
        """
        if not trade_results or len(trade_results) < 5:
            return 0.0, 0.60

        lookback = int(t["var_lookback"])
        pnl_list = []
        for tr in trade_results[-lookback:]:
            pnl = tr.get("pnl", 0.0) if isinstance(tr, dict) else 0.0
            pnl_list.append(float(pnl))

        if not pnl_list:
            return 0.0, 0.60

        arr     = np.array(pnl_list)
        var_95  = float(np.percentile(arr, 5))
        # CVaR: média dos valores abaixo do VaR
        tail    = arr[arr <= var_95]
        cvar    = float(tail.mean()) if len(tail) > 0 else var_95

        # Score: quanto melhor o CVaR, maior o score
        if cvar >= 0:    var_score = 0.90
        elif cvar > -1:  var_score = 0.75
        elif cvar > -3:  var_score = 0.58
        elif cvar > -5:  var_score = 0.42
        else:            var_score = 0.28

        return cvar, var_score

    def _check_kelly(self, signal, session, t: dict) -> float:
        """
        Verifica se o stake proposto está dentro do Kelly fraction.
        Retorna score 0–1 (1.0 = stake ótimo ou menor).
        """
        win_rate    = self._safe_float(getattr(session, "win_rate", 0.5), 0.5)
        avg_payout  = self._safe_float(getattr(session, "avg_payout",  0.85), 0.85)
        current_bal = self._safe_float(getattr(session, "current_balance", 1000.0), 1000.0)

        # Kelly fraction: f* = (p * b - q) / b
        # onde p = win_rate, q = 1-p, b = avg_payout
        p = max(0.01, min(0.99, win_rate))
        q = 1.0 - p
        b = max(0.01, avg_payout)

        kelly_f = (p * b - q) / b
        safe_f  = max(0.0, kelly_f * t["kelly_safety_factor"])

        # Stake atual (se disponível na sessão)
        current_stake = self._safe_float(getattr(session, "last_stake", 0.0))
        if current_bal <= 0 or current_stake <= 0:
            return 0.70  # sem dados suficientes

        stake_pct = current_stake / current_bal
        if safe_f <= 0:
            return 0.30  # Kelly negativo → win_rate insuficiente

        ratio = stake_pct / safe_f
        if ratio <= 1.0:   return 0.85   # dentro do Kelly
        elif ratio <= 1.5: return 0.65   # levemente acima
        elif ratio <= 2.0: return 0.45   # excedendo
        else:              return 0.25   # muito acima

    def _vol_regime_penalty(self, df: pd.DataFrame) -> float:
        """Penalidade de 0–0.2 em mercados muito voláteis."""
        try:
            if "atr_14" in df.columns and "close" in df.columns:
                atr   = self._safe_float(df["atr_14"].iloc[-1])
                price = self._safe_float(df["close"].iloc[-1])
                if price > 0:
                    atr_pct = atr / price
                    return min(0.20, max(0.0, (atr_pct - 0.005) * 10))
        except Exception:
            pass
        return 0.0

    def _compute_health(
        self, consec, drawdown, win_rate, total_trades,
        var_score, kelly_score, vol_penalty, memory_boost
    ) -> float:
        loss_penalty = min(consec / max(self._thresholds["max_consecutive_losses"], 1), 1.0) * 0.25
        dd_penalty   = min(drawdown / max(self._thresholds["max_drawdown_pct"], 0.01), 1.0) * 0.25
        wr_bonus     = max(0.0, win_rate - 0.5) * 0.30

        base = (var_score * 0.30) + (kelly_score * 0.20) + wr_bonus + 0.35
        return base - loss_penalty - dd_penalty - vol_penalty + memory_boost