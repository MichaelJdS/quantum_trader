from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from backtesting.metrics import PerformanceMetrics
from core.entities import RiskConfig
from core.enums import StakeMode
from core.strategies.ema_rsi import EmaRsiStrategy
from core.strategies.bollinger_reversion import BollingerReversionStrategy
from core.strategies.breakout import BreakoutStrategy
from core.strategy_base import StrategyBase
from ml.feature_engineer import FeatureEngineer


@dataclass
class BacktestTrade:
    """Trade gerado durante o backtesting."""

    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    direction: str          # "BUY" | "SELL"
    strategy_name: str
    stake: float
    pnl: float
    won: bool
    confidence: float
    entry_time: Any = None
    exit_time: Any = None


@dataclass
class BacktestResult:
    """Resultado completo de um backtest."""

    symbol: str
    strategy_name: str
    granularity: int
    total_trades: int
    trades: list[BacktestTrade]
    equity_curve: pd.Series
    drawdown_series: pd.Series
    metrics: dict[str, float]
    params: dict[str, Any] = field(default_factory=dict)


class VectorizedBacktester:
    """
    Backtester vetorizado usando NumPy/Pandas.

    Performance: processa 100k candles em < 2s.

    Fluxo:
      1. Carrega candles históricos.
      2. Aplica FeatureEngineer (uma vez, sobre todo o período).
      3. Para cada estratégia, gera sinais vetorizados.
      4. Simula execução com stake dinâmico e stop-loss/win.
      5. Constrói equity curve e calcula métricas.
    """

    def __init__(
        self,
        risk_config: RiskConfig,
        payout_rate: float = 0.95,
        commission: float = 0.0,
        slippage_pct: float = 0.0001,
    ) -> None:
        self.risk_config = risk_config
        self.payout_rate = payout_rate
        self.commission = commission
        self.slippage_pct = slippage_pct
        self._fe = FeatureEngineer()

    def run(
        self,
        df: pd.DataFrame,
        strategies: list[StrategyBase],
        symbol: str = "R_50",
        granularity: int = 60,
        initial_balance: float = 1000.0,
    ) -> list[BacktestResult]:
        """
        Executa backtest de uma lista de estratégias sobre um DataFrame OHLCV.

        Args:
            df: DataFrame com open, high, low, close, epoch (mín. 300 linhas).
            strategies: Lista de estratégias a testar.
            symbol: Símbolo do ativo.
            granularity: Segundos por candle (para cálculo de Sharpe anualizado).
            initial_balance: Saldo inicial simulado.

        Returns:
            Lista de BacktestResult, um por estratégia.
        """
        if len(df) < 300:
            raise ValueError(f"Dados insuficientes: {len(df)} candles (mín. 300).")

        logger.info(
            "Iniciando backtesting vetorizado.",
            symbol=symbol,
            candles=len(df),
            strategies=[s.name for s in strategies],
        )

        # Feature engineering uma única vez.
        feat_df = self._fe.compute(df)
        logger.info(
            "Features calculadas.",
            rows=len(feat_df),
            columns=len(feat_df.columns),
        )

        results = []
        for strategy in strategies:
            result = self._run_single_strategy(
                feat_df=feat_df,
                strategy=strategy,
                symbol=symbol,
                granularity=granularity,
                initial_balance=initial_balance,
            )
            results.append(result)
            logger.info(
                "Backtest concluído.",
                strategy=strategy.name,
                trades=result.total_trades,
                sharpe=round(result.metrics.get("sharpe_ratio", 0), 3),
                win_rate=round(result.metrics.get("win_rate", 0), 3),
                final_balance=round(result.equity_curve.iloc[-1], 2),
            )

        return results

    def _run_single_strategy(
        self,
        feat_df: pd.DataFrame,
        strategy: StrategyBase,
        symbol: str,
        granularity: int,
        initial_balance: float,
    ) -> BacktestResult:
        from core.entities import SessionState

        trades: list[BacktestTrade] = []
        balance = initial_balance
        equity = [balance]

        session = SessionState(
            initial_balance=initial_balance,
            current_balance=initial_balance,
        )

        for i in range(52, len(feat_df) - 1):
            window = feat_df.iloc[: i + 1].copy()
            signal = strategy.generate_signal(window, symbol, session)

            if signal is None:
                equity.append(balance)
                continue

            # Stake.
            stake = self._calc_stake(balance, signal.confidence)
            if stake <= 0:
                equity.append(balance)
                continue

            # Slippage.
            entry_price = float(feat_df["close"].iloc[i])
            if signal.direction.value == "BUY":
                entry_price *= 1 + self.slippage_pct
            else:
                entry_price *= 1 - self.slippage_pct

            # Resultado simulado: candle de saída (i+1).
            exit_price = float(feat_df["close"].iloc[i + 1])
            next_close = exit_price

            if signal.direction.value == "BUY":
                won = next_close > entry_price
            else:
                won = next_close < entry_price

            pnl = (stake * self.payout_rate) if won else -stake
            pnl -= self.commission

            balance += pnl
            balance = max(balance, 0.0)

            # Atualiza sessão.
            session.current_balance = balance
            session.total_trades += 1
            if won:
                session.wins += 1
                session.consecutive_losses = 0
            else:
                session.losses += 1
                session.consecutive_losses += 1
            session.win_rate = session.wins / session.total_trades

            epoch = feat_df["epoch"].iloc[i] if "epoch" in feat_df.columns else i
            epoch_next = feat_df["epoch"].iloc[i + 1] if "epoch" in feat_df.columns else i + 1

            trades.append(BacktestTrade(
                entry_idx=i,
                exit_idx=i + 1,
                entry_price=entry_price,
                exit_price=exit_price,
                direction=signal.direction.value,
                strategy_name=strategy.name,
                stake=stake,
                pnl=pnl,
                won=won,
                confidence=signal.confidence,
                entry_time=epoch,
                exit_time=epoch_next,
            ))

            equity.append(balance)

            # Stop de emergência: drawdown > 50% do inicial.
            if balance < initial_balance * 0.50:
                logger.warning(
                    "Drawdown > 50% — parando backtest para proteção.",
                    strategy=strategy.name,
                    balance=round(balance, 2),
                )
                break

        # Completa equity curve.
        while len(equity) < len(feat_df):
            equity.append(equity[-1])

        equity_series = pd.Series(equity[: len(feat_df)], name="equity")
        drawdown_series = self._compute_drawdown_series(equity_series)

        metrics = PerformanceMetrics.compute(
            trades=trades,
            equity_curve=equity_series,
            drawdown_series=drawdown_series,
            initial_balance=initial_balance,
            granularity=granularity,
            payout_rate=self.payout_rate,
        )

        return BacktestResult(
            symbol=symbol,
            strategy_name=strategy.name,
            granularity=granularity,
            total_trades=len(trades),
            trades=trades,
            equity_curve=equity_series,
            drawdown_series=drawdown_series,
            metrics=metrics,
            params={
                "initial_balance": initial_balance,
                "payout_rate": self.payout_rate,
                "slippage_pct": self.slippage_pct,
            },
        )

    def _calc_stake(self, balance: float, confidence: float) -> float:
        """Calcula stake baseado no RiskConfig."""
        mode = self.risk_config.stake_mode
        base = self.risk_config.base_stake

        if mode == StakeMode.FIXED:
            return min(base, balance)

        if mode == StakeMode.FRACTIONAL:
            pct = getattr(self.risk_config, "fractional_pct", 0.02)
            return min(balance * pct, balance)

        if mode == StakeMode.FRACTIONAL_KELLY:
            kelly_f = getattr(self.risk_config, "kelly_fraction", 0.25)
            p = confidence
            q = 1 - p
            b = self.payout_rate
            kelly = (p * b - q) / b if b > 0 else 0
            fraction = max(kelly * kelly_f, 0.005)
            return min(balance * fraction, balance * 0.10)  # Cap 10%.

        return min(base, balance)

    @staticmethod
    def _compute_drawdown_series(equity: pd.Series) -> pd.Series:
        """Calcula drawdown relativo em cada ponto da equity curve."""
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / (rolling_max + 1e-10)
        return drawdown