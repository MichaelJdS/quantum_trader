#!/usr/bin/env python3
"""
Quantum Trader — Ponto de entrada principal.

Uso:
    python main.py --symbols R_50 R_75 --dry-run
    python main.py --symbols R_50 --live --granularity 60
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid

from loguru import logger


async def run(args: argparse.Namespace) -> None:
    from core.bootstrap import bootstrap
    await bootstrap()

    from core.entities import RiskConfig
    from core.enums import StakeMode
    from core.execution_engine import ExecutionEngine
    from core.settings import get_settings
    from core.strategies import BollingerReversionStrategy, BreakoutStrategy, EmaRsiStrategy
    from infra.deriv_client import DerivClient
    from infra.symbol_manager import SymbolManager
    from tui.app import QuantumTraderApp
    from tui.state import TUIState

    settings = get_settings()
    dry_run = not args.live

    # 1. Configura risco.
    risk_config = RiskConfig(
        stake_mode=StakeMode.FRACTIONAL_KELLY,
        base_stake=args.stake,
        stop_win_pct=args.stop_win / 100,
        stop_loss_pct=args.stop_loss / 100,
        max_daily_drawdown_pct=args.max_drawdown / 100,
        max_consecutive_losses=args.max_losses,
        kelly_fraction=0.25,
    )

    # 2. Conecta ao Deriv.
    client = DerivClient(dry_run=dry_run)
    await client.connect()

    # 3. Inicializa símbolos.
    symbols = args.symbols
    sm = SymbolManager(
        client=client,
        symbols=symbols,
        granularity=args.granularity,
    )
    await sm.initialize_all()

    # 4. Engine + estratégias.
    session_id = str(uuid.uuid4())
    engine = ExecutionEngine(
        client=client,
        symbol_manager=sm,
        risk_config=risk_config,
        session_id=session_id,
        dry_run=dry_run,
    )
    engine.register_strategy(EmaRsiStrategy(risk_config=risk_config))
    engine.register_strategy(BollingerReversionStrategy(risk_config=risk_config))
    engine.register_strategy(BreakoutStrategy(risk_config=risk_config))
    await engine.start()

    # 5. Estado da TUI.
    state = TUIState(
        session_id=session_id,
        dry_run=dry_run,
        initial_balance=engine.session_state.initial_balance,
        balance=engine.session_state.initial_balance,
        active_symbols=symbols,
    )

    # 6. Inicia TUI.
    logger.info("Iniciando TUI.", symbols=symbols, dry_run=dry_run)
    app = QuantumTraderApp(engine=engine, state=state)
    await app.run_async()

    # 7. Cleanup.
    await engine.stop()
    await client.disconnect()
    logger.info("Quantum Trader encerrado.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantum Trader — Deriv Bot")
    parser.add_argument("--symbols", nargs="+", default=["R_50", "R_75"])
    parser.add_argument("--live", action="store_true", help="Modo live (default: dry-run)")
    parser.add_argument("--granularity", type=int, default=60, help="Segundos por candle")
    parser.add_argument("--stake", type=float, default=1.0, help="Stake base")
    parser.add_argument("--stop-win", type=float, default=5.0, help="Stop Win %% da banca")
    parser.add_argument("--stop-loss", type=float, default=3.0, help="Stop Loss %% da banca")
    parser.add_argument("--max-drawdown", type=float, default=5.0, help="Max drawdown diário %%")
    parser.add_argument("--max-losses", type=int, default=5, help="Max perdas consecutivas")
    args = parser.parse_args()

    if args.live:
        print("\n⚠️  MODO LIVE ATIVADO — operações REAIS serão executadas!")
        print("    Pressione ENTER para continuar ou CTRL+C para cancelar.\n")
        try:
            input()
        except KeyboardInterrupt:
            print("Cancelado.")
            sys.exit(0)

    asyncio.run(run(args))


if __name__ == "__main__":
    main()