from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from backtesting.vectorized_backtester import BacktestTrade


class PerformanceMetrics:
    """
    Cálculo de métricas de performance para estratégias de trading.

    Métricas calculadas:
      Retorno     : total_return, annualized_return, final_balance.
      Risco       : max_drawdown, avg_drawdown, volatility, VaR_95, CVaR_95.
      Risco/Ret   : sharpe_ratio, sortino_ratio, calmar_ratio, omega_ratio.
      Trading     : win_rate, profit_factor, avg_win, avg_loss, payoff_ratio.
      Consistência: recovery_factor, ulcer_index, expectancy.
    """

    # Candles por ano por granularidade.
    CANDLES_PER_YEAR: dict[int, int] = {
        1: 365 * 24 * 3600,
        60: 365 * 24 * 60,
        300: 365 * 24 * 12,
        3600: 365 * 24,
        86400: 252,
    }

    @classmethod
    def compute(
        cls,
        trades: list["BacktestTrade"],
        equity_curve: pd.Series,
        drawdown_series: pd.Series,
        initial_balance: float,
        granularity: int = 60,
        payout_rate: float = 0.95,
    ) -> dict[str, float]:
        """
        Calcula todas as métricas de performance.

        Returns:
            dict com todas as métricas. Valores NaN substituídos por 0.0.
        """
        metrics: dict[str, float] = {}

        # ── Retorno ────────────────────────────────────────────────────────
        final = float(equity_curve.iloc[-1])
        metrics["final_balance"] = round(final, 4)
        metrics["total_return"] = round((final - initial_balance) / (initial_balance + 1e-10), 6)
        metrics["total_return_pct"] = round(metrics["total_return"] * 100, 4)

        candles_per_year = cls.CANDLES_PER_YEAR.get(granularity, 365 * 24 * 60)
        n_candles = len(equity_curve)
        years = n_candles / (candles_per_year + 1e-10)

        if years > 0 and metrics["total_return"] > -1:
            annualized = (1 + metrics["total_return"]) ** (1 / years) - 1
        else:
            annualized = 0.0
        metrics["annualized_return"] = round(annualized, 6)

        # ── Drawdown ───────────────────────────────────────────────────────
        dd = drawdown_series.values
        metrics["max_drawdown"] = round(float(np.min(dd)), 6)
        metrics["max_drawdown_pct"] = round(abs(metrics["max_drawdown"]) * 100, 4)
        metrics["avg_drawdown"] = round(float(np.mean(dd[dd < 0])) if any(dd < 0) else 0.0, 6)
        metrics["recovery_factor"] = round(
            abs(metrics["total_return"]) / (abs(metrics["max_drawdown"]) + 1e-10), 4
        )
        metrics["ulcer_index"] = round(
            float(np.sqrt(np.mean(dd ** 2))), 6
        )

        # ── Volatilidade ───────────────────────────────────────────────────
        returns = equity_curve.pct_change().dropna().values
        if len(returns) > 1:
            vol = float(np.std(returns)) * math.sqrt(candles_per_year)
            metrics["annualized_volatility"] = round(vol, 6)

            downside_returns = returns[returns < 0]
            downside_vol = (
                float(np.std(downside_returns)) * math.sqrt(candles_per_year)
                if len(downside_returns) > 1 else 1e-10
            )
            metrics["downside_volatility"] = round(downside_vol, 6)
        else:
            metrics["annualized_volatility"] = 0.0
            metrics["downside_volatility"] = 0.0

        # ── VaR / CVaR ─────────────────────────────────────────────────────
        if len(returns) > 10:
            metrics["var_95"] = round(float(np.percentile(returns, 5)), 6)
            cvar_mask = returns <= metrics["var_95"]
            metrics["cvar_95"] = round(
                float(np.mean(returns[cvar_mask])) if cvar_mask.any() else 0.0, 6
            )
        else:
            metrics["var_95"] = 0.0
            metrics["cvar_95"] = 0.0

        # ── Sharpe / Sortino / Calmar ──────────────────────────────────────
        risk_free = 0.0  # Binários: sem taxa livre de risco relevante.
        ann_ret = metrics["annualized_return"]
        vol_ann = metrics["annualized_volatility"]
        down_vol = metrics["downside_volatility"]
        max_dd = abs(metrics["max_drawdown"])

        metrics["sharpe_ratio"] = round(
            (ann_ret - risk_free) / (vol_ann + 1e-10), 4
        )
        metrics["sortino_ratio"] = round(
            (ann_ret - risk_free) / (down_vol + 1e-10), 4
        )
        metrics["calmar_ratio"] = round(
            ann_ret / (max_dd + 1e-10), 4
        )

        # ── Omega Ratio ────────────────────────────────────────────────────
        threshold = 0.0
        gains = returns[returns > threshold]
        losses = returns[returns <= threshold]
        metrics["omega_ratio"] = round(
            float(gains.sum() / (abs(losses.sum()) + 1e-10)), 4
        ) if len(losses) > 0 else float("inf")

        # ── Métricas de Trading ────────────────────────────────────────────
        if trades:
            pnls = np.array([t.pnl for t in trades])
            wins = [t for t in trades if t.won]
            losses_t = [t for t in trades if not t.won]

            metrics["total_trades"] = len(trades)
            metrics["win_rate"] = round(len(wins) / len(trades), 6)
            metrics["loss_rate"] = round(len(losses_t) / len(trades), 6)

            avg_win = float(np.mean([t.pnl for t in wins])) if wins else 0.0
            avg_loss = abs(float(np.mean([t.pnl for t in losses_t]))) if losses_t else 1e-10
            metrics["avg_win"] = round(avg_win, 4)
            metrics["avg_loss"] = round(-avg_loss, 4)
            metrics["payoff_ratio"] = round(avg_win / (avg_loss + 1e-10), 4)

            gross_profit = sum(t.pnl for t in wins)
            gross_loss = abs(sum(t.pnl for t in losses_t))
            metrics["profit_factor"] = round(
                gross_profit / (gross_loss + 1e-10), 4
            )

            # Expectância por trade.
            p_win = metrics["win_rate"]
            p_loss = 1 - p_win
            metrics["expectancy"] = round(
                p_win * avg_win - p_loss * avg_loss, 6
            )
            metrics["expectancy_pct"] = round(
                metrics["expectancy"] / (
                    float(np.mean([t.stake for t in trades])) + 1e-10
                ) * 100, 4
            )

            # Maior série de perdas consecutivas.
            consec = cls._max_consecutive_losses(trades)
            metrics["max_consecutive_losses"] = consec

            # Média de stake.
            metrics["avg_stake"] = round(float(np.mean([t.stake for t in trades])), 4)

            # Total de PnL.
            metrics["total_pnl"] = round(float(pnls.sum()), 4)
            metrics["total_pnl_pct"] = round(
                float(pnls.sum()) / (initial_balance + 1e-10) * 100, 4
            )
        else:
            for k in [
                "total_trades", "win_rate", "loss_rate", "avg_win", "avg_loss",
                "payoff_ratio", "profit_factor", "expectancy", "expectancy_pct",
                "max_consecutive_losses", "avg_stake", "total_pnl", "total_pnl_pct",
            ]:
                metrics[k] = 0.0

        # Limpa NaN e inf.
        for k, v in metrics.items():
            if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                metrics[k] = 0.0

        return metrics

    @staticmethod
    def _max_consecutive_losses(trades: list["BacktestTrade"]) -> int:
        max_seq = 0
        current = 0
        for t in trades:
            if not t.won:
                current += 1
                max_seq = max(max_seq, current)
            else:
                current = 0
        return max_seq

    @classmethod
    def compare_strategies(
        cls,
        results: list,
        sort_by: str = "sharpe_ratio",
    ) -> pd.DataFrame:
        """
        Gera tabela comparativa de múltiplas estratégias.

        Args:
            results: Lista de BacktestResult.
            sort_by: Métrica para ordenação.

        Returns:
            DataFrame com métricas principais por estratégia.
        """
        KEY_METRICS = [
            "total_return_pct", "sharpe_ratio", "sortino_ratio", "calmar_ratio",
            "win_rate", "profit_factor", "expectancy_pct", "max_drawdown_pct",
            "total_trades", "max_consecutive_losses", "omega_ratio",
        ]
        rows = []
        for r in results:
            row = {"strategy": r.strategy_name, "symbol": r.symbol}
            for m in KEY_METRICS:
                row[m] = r.metrics.get(m, 0.0)
            rows.append(row)

        compare_df = pd.DataFrame(rows)
        if sort_by in compare_df.columns:
            compare_df = compare_df.sort_values(sort_by, ascending=False)
        return compare_df.reset_index(drop=True)