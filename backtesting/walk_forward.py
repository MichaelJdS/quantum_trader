from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from loguru import logger

from backtesting.metrics import PerformanceMetrics
from backtesting.vectorized_backtester import BacktestResult, VectorizedBacktester
from core.entities import RiskConfig
from core.strategy_base import StrategyBase
from ml.mlops import MLOpsManager


@dataclass
class WalkForwardResult:
    """Resultado do Walk-Forward com métricas por fold e agregadas."""

    strategy_name: str
    symbol: str
    n_folds: int
    fold_results: list[BacktestResult]
    aggregated_metrics: dict[str, float]
    stability_score: float  # Consistência entre folds (0-1).


class WalkForwardValidator:
    """
    Validação Walk-Forward com Purge Gap para séries temporais.

    Divide o histórico em N folds:
      [===TRAIN===][GAP][==VAL==] [===TRAIN===][GAP][==VAL==] ...

    GAP (purge): evita data leakage entre treino e validação.

    Métricas agregadas: média ponderada por número de trades.
    Stability Score: coeficiente de variação inverso do Sharpe entre folds.
    """

    def __init__(
        self,
        risk_config: RiskConfig,
        n_folds: int = 5,
        purge_gap: int = 20,
        train_ratio: float = 0.70,
        payout_rate: float = 0.95,
    ) -> None:
        self.risk_config = risk_config
        self.n_folds = n_folds
        self.purge_gap = purge_gap
        self.train_ratio = train_ratio
        self.payout_rate = payout_rate

    def validate(
        self,
        df: pd.DataFrame,
        strategy: StrategyBase,
        symbol: str = "R_50",
        initial_balance: float = 1000.0,
    ) -> WalkForwardResult:
        """
        Executa Walk-Forward para uma estratégia.
        """
        n = len(df)
        fold_size = n // (self.n_folds + 1)
        fold_results: list[BacktestResult] = []

        backtester = VectorizedBacktester(
            risk_config=self.risk_config,
            payout_rate=self.payout_rate,
        )

        for fold in range(self.n_folds):
            train_start = fold * fold_size
            train_end = train_start + int(fold_size * (self.train_ratio + 1))
            val_start = train_end + self.purge_gap
            val_end = val_start + fold_size

            if val_end > n:
                break

            val_df = df.iloc[val_start:val_end].copy().reset_index(drop=True)

            if len(val_df) < 200:
                logger.debug(f"Fold {fold + 1}: dados insuficientes, pulando.")
                continue

            logger.info(
                f"Walk-Forward fold {fold + 1}/{self.n_folds}.",
                val_rows=len(val_df),
                strategy=strategy.name,
            )

            try:
                results = backtester.run(
                    df=val_df,
                    strategies=[strategy],
                    symbol=symbol,
                    initial_balance=initial_balance,
                )
                if results:
                    fold_results.append(results[0])
            except Exception as exc:
                logger.error(f"Fold {fold + 1} falhou.", error=str(exc))
                continue

        aggregated = self._aggregate_metrics(fold_results)
        stability = self._compute_stability(fold_results)

        logger.info(
            "Walk-Forward concluído.",
            strategy=strategy.name,
            folds_completed=len(fold_results),
            sharpe_mean=round(aggregated.get("sharpe_ratio", 0), 3),
            stability=round(stability, 3),
        )

        return WalkForwardResult(
            strategy_name=strategy.name,
            symbol=symbol,
            n_folds=len(fold_results),
            fold_results=fold_results,
            aggregated_metrics=aggregated,
            stability_score=stability,
        )

    def _aggregate_metrics(
        self, results: list[BacktestResult]
    ) -> dict[str, float]:
        """Média ponderada por número de trades."""
        if not results:
            return {}

        all_metrics: dict[str, list[float]] = {}
        weights: list[float] = []

        for r in results:
            w = max(r.total_trades, 1)
            weights.append(w)
            for k, v in r.metrics.items():
                all_metrics.setdefault(k, []).append(v)

        total_w = sum(weights)
        aggregated = {}
        for k, vals in all_metrics.items():
            weighted_avg = sum(v * w for v, w in zip(vals, weights)) / total_w
            aggregated[k] = round(weighted_avg, 6)

        return aggregated

    def _compute_stability(self, results: list[BacktestResult]) -> float:
        """
        Stability Score: 1 - CV(Sharpe).

        Quanto mais consistente o Sharpe entre folds, mais próximo de 1.
        """
        if len(results) < 2:
            return 0.0
        sharpes = [r.metrics.get("sharpe_ratio", 0.0) for r in results]
        mean = sum(sharpes) / len(sharpes)
        if abs(mean) < 1e-10:
            return 0.0
        std = (sum((s - mean) ** 2 for s in sharpes) / len(sharpes)) ** 0.5
        cv = std / abs(mean)
        return round(max(0.0, 1 - cv), 4)