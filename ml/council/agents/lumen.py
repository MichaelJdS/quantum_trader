"""
ml/council/agents/lumen.py — LUMEN: Análise de Correlação Cross-Asset & Anomalias

Compara o comportamento de R_50, R_75 e R_100 simultaneamente.
Detecta divergências anômalas entre os índices sintéticos.
Quando todos se movem na mesma direção → confirmação de sinal.
Quando divergem → cautela.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.council.base_agent import AgentVote, BaseAgent


class LumenAgent(BaseAgent):
    """Especialista em Correlação Cross-Asset — LUMEN."""

    name = "LUMEN"
    weight = 0.05

    CORR_WINDOW   = 20   # períodos para correlação rolling
    ANOMALY_STD   = 2.0  # desvios padrão para anomalia de volatilidade

    def analyze(self, signal, df, session, ticks=None, peer_dfs=None) -> AgentVote:
        sig_dir = signal.direction.value if hasattr(signal.direction, "value") else str(signal.direction)
        is_buy  = sig_dir in ("BUY", "buy", "CALL", "call")

        if peer_dfs is None or len(peer_dfs) < 1:
            # Sem dados de outros símbolos → análise apenas interna
            return self._analyze_single(df, is_buy)

        return self._analyze_cross_asset(df, peer_dfs, is_buy, signal.symbol)

    # ── Análise cross-asset ───────────────────────────────────────────────────

    def _analyze_cross_asset(
        self,
        df: pd.DataFrame,
        peer_dfs: dict[str, pd.DataFrame],
        is_buy: bool,
        symbol: str,
    ) -> AgentVote:
        # Coleta retornos recentes de cada símbolo
        all_returns: dict[str, np.ndarray] = {}

        for sym, sym_df in {symbol: df, **peer_dfs}.items():
            if len(sym_df) >= self.CORR_WINDOW + 1:
                closes = sym_df["close"].tail(self.CORR_WINDOW + 1).astype(float)
                rets   = closes.pct_change().dropna().values
                all_returns[sym] = rets

        if len(all_returns) < 2:
            return self._analyze_single(df, is_buy)

        # Direção dos retornos na última vela
        last_directions = {}
        for sym, rets in all_returns.items():
            last_directions[sym] = "UP" if rets[-1] > 0 else "DOWN"

        up_count   = sum(1 for d in last_directions.values() if d == "UP")
        down_count = sum(1 for d in last_directions.values() if d == "DOWN")
        total      = len(last_directions)

        # Correlação média entre pares
        syms   = list(all_returns.keys())
        corrs  = []
        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                r1 = all_returns[syms[i]]
                r2 = all_returns[syms[j]]
                n  = min(len(r1), len(r2))
                if n >= 5:
                    corr = float(np.corrcoef(r1[-n:], r2[-n:])[0, 1])
                    if not np.isnan(corr):
                        corrs.append(corr)

        avg_corr = float(np.mean(corrs)) if corrs else 0.5

        # ── Anomalia de volatilidade ──────────────────────────────────────────
        vols = {}
        for sym, rets in all_returns.items():
            vols[sym] = float(np.std(rets))

        if vols:
            vol_mean = np.mean(list(vols.values()))
            vol_std  = np.std(list(vols.values()))
            vol_anomaly = any(
                abs(v - vol_mean) > self.ANOMALY_STD * vol_std
                for v in vols.values()
            ) if vol_std > 0 else False
        else:
            vol_anomaly = False

        # ── Decisão ───────────────────────────────────────────────────────────
        if vol_anomaly:
            return AgentVote(
                self.name, "NEUTRAL", 0.35,
                reasoning=f"Anomalia de volatilidade cross-asset detectada (corr={avg_corr:.2f})"
            )

        consensus_up   = up_count / total if total else 0.5
        consensus_down = down_count / total if total else 0.5

        if avg_corr > 0.7:
            # Alta correlação + consenso direcional = confirmação forte
            if consensus_up > 0.7 and is_buy:
                return AgentVote(self.name, "BUY", 0.72,
                                 reasoning=f"Consenso cross-asset UP ({up_count}/{total}, corr={avg_corr:.2f})")
            elif consensus_down > 0.7 and not is_buy:
                return AgentVote(self.name, "SELL", 0.72,
                                 reasoning=f"Consenso cross-asset DOWN ({down_count}/{total}, corr={avg_corr:.2f})")

        if consensus_up > 0.6:
            return AgentVote(self.name, "BUY", 0.60,
                             reasoning=f"Maioria cross-asset UP ({up_count}/{total})")
        elif consensus_down > 0.6:
            return AgentVote(self.name, "SELL", 0.60,
                             reasoning=f"Maioria cross-asset DOWN ({down_count}/{total})")

        return AgentVote(self.name, "NEUTRAL", 0.5,
                         reasoning=f"Divergência cross-asset (up={up_count} down={down_count})")

    # ── Fallback: análise interna ─────────────────────────────────────────────

    def _analyze_single(self, df: pd.DataFrame, is_buy: bool) -> AgentVote:
        """Sem peers: analisa volatilidade do próprio ativo."""
        if len(df) < self.CORR_WINDOW:
            return AgentVote(self.name, "NEUTRAL", 0.5, reasoning="Sem dados de peers")

        closes = df["close"].tail(self.CORR_WINDOW).astype(float)
        rets   = closes.pct_change().dropna()

        vol_recent  = float(rets.tail(5).std()) if len(rets) >= 5 else 0.0
        vol_hist    = float(rets.std()) if len(rets) >= 10 else vol_recent

        if vol_hist > 0 and vol_recent > vol_hist * 2:
            return AgentVote(self.name, "NEUTRAL", 0.40,
                             reasoning=f"Vol. recente ({vol_recent:.4f}) muito acima da histórica ({vol_hist:.4f})")

        action = "BUY" if is_buy else "SELL"
        return AgentVote(self.name, action, 0.55,
                         reasoning=f"Vol. normal (rec={vol_recent:.4f} hist={vol_hist:.4f})")
