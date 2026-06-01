"""
CLI de Backtesting do Quantum Trader.

Uso:
    python scripts/run_backtest.py --symbol R_50 --candles 3000 --all-strategies
    python scripts/run_backtest.py --symbol R_75 --strategy ema_rsi --walk-forward
    python scripts/run_backtest.py --symbol R_50 --event-driven --candles 1000
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def main(args: argparse.Namespace) -> None:
    from core.bootstrap import bootstrap
    await bootstrap()

    from core.entities import RiskConfig
    from core.enums import StakeMode
    from core.strategies import BollingerReversionStrategy, BreakoutStrategy, EmaRsiStrategy
    from infra.db.database import get_session
    from infra.db.repository import CandleRepository

    risk_config = RiskConfig(
        stake_mode=StakeMode.FRACTIONAL_KELLY,
        base_stake=1.0,
        stop_win_pct=0.05,
        stop_loss_pct=0.03,
        max_daily_drawdown_pct=0.05,
        max_consecutive_losses=5,
        kelly_fraction=0.25,
    )

    # 1. Carrega candles.
    async with get_session() as db:
        repo = CandleRepository(db)
        candle_rows = await repo.get_recent(
            args.symbol, args.granularity, limit=args.candles
        )

    if len(candle_rows) < 300:
        print(f"❌ Candles insuficientes: {len(candle_rows)}.")
        return

    import pandas as pd
    df = pd.DataFrame([{
        "open": c.open, "high": c.high, "low": c.low,
        "close": c.close, "epoch": c.epoch,
    } for c in candle_rows])
    print(f"✅ {len(df)} candles carregados para {args.symbol}.")

    # 2. Estratégias.
    all_strategies = {
        "ema_rsi": EmaRsiStrategy(risk_config=risk_config),
        "bollinger": BollingerReversionStrategy(risk_config=risk_config),
        "breakout": BreakoutStrategy(risk_config=risk_config),
    }
    if args.all_strategies:
        strategies = list(all_strategies.values())
    elif args.strategy:
        strategies = [all_strategies[args.strategy]]
    else:
        strategies = list(all_strategies.values())

    # 3. Backtesting.
    if args.event_driven:
        from backtesting.event_driven_backtester import EventDrivenBacktester
        bt = EventDrivenBacktester(risk_config=risk_config)
        results = [
            bt.run(df, strategy=s, symbol=args.symbol,
                   initial_balance=args.balance)
            for s in strategies
        ]
    elif args.walk_forward:
        from backtesting.walk_forward import WalkForwardValidator
        wfv = WalkForwardValidator(risk_config=risk_config, n_folds=5)
        wf_results = [
            wfv.validate(df, strategy=s, symbol=args.symbol,
                         initial_balance=args.balance)
            for s in strategies
        ]
        results = [wfr.fold_results[0] for wfr in wf_results if wfr.fold_results]

        print("\n📊 Walk-Forward Results:")
        for wfr in wf_results:
            print(
                f"  {wfr.strategy_name}: Sharpe={wfr.aggregated_metrics.get('sharpe_ratio', 0):.3f}"
                f" | Stability={wfr.stability_score:.3f}"
                f" | Folds={wfr.n_folds}"
            )
    else:
        from backtesting.vectorized_backtester import VectorizedBacktester
        bt = VectorizedBacktester(risk_config=risk_config)
        results = bt.run(df, strategies=strategies, symbol=args.symbol,
                         initial_balance=args.balance)

    if not results:
        print("❌ Nenhum resultado gerado.")
        return

    # 4. Relatório HTML.
    from backtesting.report_generator import ReportGenerator
    from backtesting.metrics import PerformanceMetrics

    gen = ReportGenerator(output_dir="./reports")
    report_path = gen.generate(results, title=f"Backtest — {args.symbol}")
    print(f"\n✅ Relatório gerado: {report_path}")

    # 5. Comparativo no terminal.
    compare_df = PerformanceMetrics.compare_strategies(results)
    print("\n📊 Comparativo de Estratégias:")
    print(compare_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtesting do Quantum Trader.")
    parser.add_argument("--symbol", default="R_50")
    parser.add_argument("--candles", type=int, default=3000)
    parser.add_argument("--granularity", type=int, default=60)
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--strategy", choices=["ema_rsi", "bollinger", "breakout"])
    parser.add_argument("--all-strategies", action="store_true")
    parser.add_argument("--event-driven", action="store_true")
    parser.add_argument("--walk-forward", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args))