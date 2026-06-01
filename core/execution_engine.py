from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone

from loguru import logger

from core.entities import RiskConfig, SessionState, Signal, Trade
from core.enums import ContractType, TradeDirection, TradeStatus
from core.exceptions import (
    RiskViolationError,
    StopLossReachedError,
    StopWinReachedError,
)
from core.risk_manager import RiskManager
from core.strategy_base import StrategyBase
from infra.cache import FeatureCache
from infra.db.database import get_session
from infra.db.repository import SessionRepository, StopEventRepository, TradeRepository
from infra.deriv_client import DerivClient
from infra.symbol_manager import SymbolManager
from ml.feature_engineer import FeatureEngineer


@dataclass
class ExecutionEngine:
    """
    Orquestrador central de execução do Quantum Trader.

    Fluxo por tick:
      1. Obtém DataFrame de candles do SymbolManager.
      2. Aplica FeatureEngineer ao DataFrame.
      3. Para cada estratégia registrada, gera sinal.
      4. Valida risco via RiskManager.
      5. Obtém proposta da API Deriv.
      6. Executa contrato (ou simula em dry-run).
      7. Persiste trade no banco.
      8. Aguarda resultado e atualiza banco + métricas.
    """

    client: DerivClient
    symbol_manager: SymbolManager
    risk_config: RiskConfig
    session_id: str
    dry_run: bool = True

    _strategies: list[StrategyBase] = field(default_factory=list, init=False)
    _risk_manager: RiskManager = field(init=False)
    _feature_engineer: FeatureEngineer = field(init=False)
    _session_state: SessionState = field(init=False)
    _open_trades: dict[str, Trade] = field(default_factory=dict, init=False)
    # FIX B9: Mantém referência forte às tasks para evitar GC prematuro.
    _pending_tasks: set[asyncio.Task] = field(default_factory=set, init=False)
    # FIX C1: Mapa de contract_id → Future para receber resultado live.
    _contract_futures: dict[str, asyncio.Future] = field(default_factory=dict, init=False)
    _running: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._feature_engineer = FeatureEngineer()
        self._risk_manager = RiskManager(
            config=self.risk_config,
            initial_balance=0.0,  # Atualizado no start().
        )
        self._session_state = SessionState(
            initial_balance=0.0,
            current_balance=0.0,
        )

    # ── Setup ─────────────────────────────────────────────────────────────────

    def register_strategy(self, strategy: StrategyBase) -> None:
        """Adiciona estratégia ao pool de execução."""
        self._strategies.append(strategy)
        logger.info("Estratégia registrada.", strategy=strategy.name)

    async def start(self) -> None:
        """
        Inicializa o engine:
          1. Carrega saldo atual da conta.
          2. Registra tick listeners para todos os símbolos.
          3. Registra listener de proposal_open_contract para modo live.
        """
        balance_data = await self.client.get_balance()
        balance = float(balance_data.get("balance", 1000.0))

        self._risk_manager = RiskManager(
            config=self.risk_config,
            initial_balance=balance,
        )
        self._session_state = SessionState(
            initial_balance=balance,
            current_balance=balance,
        )

        for symbol in self.symbol_manager.ready_symbols:
            self.symbol_manager.add_tick_listener(
                symbol,
                self._make_tick_handler(symbol),
            )

        # FIX C1: Registra listener para resultados reais de contratos (modo live).
        if not self.dry_run:
            self.client.on("proposal_open_contract", self._handle_contract_update)

        self._running = True
        logger.success(
            "ExecutionEngine iniciado.",
            balance=balance,
            symbols=self.symbol_manager.ready_symbols,
            strategies=[s.name for s in self._strategies],
            dry_run=self.dry_run,
        )

    async def stop(self) -> None:
        """Para o engine, cancela tasks pendentes e persiste métricas finais."""
        self._running = False

        # FIX B22: Cancela todas as tasks de _await_result ainda pendentes
        # e marca os trades abertos como CANCELLED no banco.
        if self._pending_tasks:
            logger.info(
                "Cancelando tasks pendentes.",
                count=len(self._pending_tasks),
            )
            for task in list(self._pending_tasks):
                task.cancel()

        if self._open_trades:
            closed_at = datetime.now(tz=timezone.utc)
            async with get_session() as db:
                repo = TradeRepository(db)
                for trade in self._open_trades.values():
                    await repo.update_result(
                        trade_id=trade.id,
                        status=TradeStatus.CANCELLED,
                        exit_price=None,
                        pnl=0.0,
                        payout=None,
                        closed_at=closed_at,
                    )
                    logger.warning(
                        "Trade cancelado no shutdown.",
                        trade_id=trade.id,
                        symbol=trade.symbol,
                    )
            self._open_trades.clear()

        # Cancela futures de contratos live pendentes.
        for fut in self._contract_futures.values():
            if not fut.done():
                fut.cancel()
        self._contract_futures.clear()

        metrics = self._risk_manager.session_metrics(self._session_state)
        async with get_session() as db:
            repo = SessionRepository(db)
            await repo.close_session(
                session_id=self.session_id,
                final_balance=self._session_state.current_balance,
                metrics=metrics,
            )

        logger.info(
            "ExecutionEngine parado.",
            metrics=metrics,
        )

    # ── Tick Handler ──────────────────────────────────────────────────────────

    def _make_tick_handler(self, symbol: str):
        """Factory de handler assíncrono por símbolo."""
        async def handler(data: dict) -> None:
            if not self._running:
                return
            await self._process_symbol(symbol)
        return handler

    async def _process_symbol(self, symbol: str) -> None:
        """Pipeline completo para um símbolo em um tick."""
        # 1. Candles + features.
        raw_df = self.symbol_manager.get_candles_df(symbol)
        if raw_df.empty or len(raw_df) < 55:
            return

        cache = FeatureCache.get_instance()
        cached_df = cache.get_features(symbol, "features_df")
        last_epoch = int(raw_df.iloc[-1]["epoch"])

        # FIX B17: Invalida por epoch E por TTL (evita cache stale pós-reconexão).
        if (
            cached_df is not None
            and int(cached_df.iloc[-1].get("epoch", 0)) == last_epoch
            and not cache.is_expired(symbol, "features_df", ttl_seconds=30)
        ):
            feat_df = cached_df
        else:
            feat_df = self._feature_engineer.compute(raw_df)
            cache.set_features(symbol, "features_df", feat_df)

        if feat_df.empty:
            return

        # 2. Avalia cada estratégia.
        for strategy in self._strategies:
            signal = strategy.generate_signal(feat_df, symbol, self._session_state)
            if signal is None:
                continue
            await self._handle_signal(signal)
            break  # Uma operação por tick por símbolo.

    # ── Signal → Execution ────────────────────────────────────────────────────

    async def _handle_signal(self, signal: Signal) -> None:
        """Valida risco, executa e persiste o trade."""

        # 1. Verifica se pode operar.
        try:
            self._risk_manager.assert_can_trade(self._session_state)
        except StopWinReachedError as exc:
            logger.success("Stop Win atingido — encerrando sessão.", reason=str(exc))
            await self._record_stop_event("WIN", str(exc))
            self._running = False
            return
        except StopLossReachedError as exc:
            logger.warning("Stop Loss atingido — encerrando sessão.", reason=str(exc))
            await self._record_stop_event("LOSS", str(exc))
            self._running = False
            return
        except RiskViolationError as exc:
            logger.warning("Operação bloqueada por risco.", reason=str(exc))
            return

        # 2. Calcula stake.
        stake = self._risk_manager.calculate_stake(
            current_balance=self._session_state.current_balance,
            win_probability=signal.confidence,
        )

        # 3. Proposta Deriv.
        try:
            proposal = await self.client.get_proposal(
                symbol=signal.symbol,
                contract_type=signal.contract_type.value,
                duration=self._get_duration(signal.strategy_name),
                duration_unit=self._get_duration_unit(signal.strategy_name),
                amount=stake,
            )
        except Exception as exc:
            logger.error("Falha ao obter proposta.", error=str(exc))
            return

        # FIX B14: Valida proposal_id antes de prosseguir.
        proposal_id = proposal.get("id", "")
        if not proposal_id:
            logger.error(
                "Proposta retornou sem ID — operação abortada.",
                symbol=signal.symbol,
                proposal=proposal,
            )
            return

        ask_price = float(proposal.get("ask_price", stake))

        # 4. Cria entidade Trade.
        trade = Trade(
            symbol=signal.symbol,
            stake=ask_price,
            contract_type=signal.contract_type,
            direction=signal.direction,
            status=TradeStatus.OPEN,
            confidence=signal.confidence,
            strategy_name=signal.strategy_name,
            model_name=signal.model_name,
            entry_price=signal.entry_price,
            opened_at=datetime.now(tz=timezone.utc),
        )

        # 5. Persiste antes de executar.
        async with get_session() as db:
            repo = TradeRepository(db)
            await repo.save(trade, self.session_id)

        # 6. Executa.
        try:
            result = await self.client.buy_contract(
                proposal_id=proposal_id,
                price=ask_price,
            )
            contract_id = result.get("buy", {}).get("contract_id", "")
            self._open_trades[trade.id] = trade
            logger.success(
                "Contrato aberto.",
                trade_id=trade.id,
                symbol=trade.symbol,
                contract_id=contract_id,
                stake=ask_price,
                dry_run=self.dry_run,
            )
        except Exception as exc:
            logger.error("Falha ao executar contrato.", error=str(exc))
            async with get_session() as db:
                repo = TradeRepository(db)
                await repo.update_result(
                    trade_id=trade.id,
                    status=TradeStatus.CANCELLED,
                    exit_price=None,
                    pnl=0.0,
                    payout=None,
                    closed_at=datetime.now(tz=timezone.utc),
                )
            return

        # FIX C1: Em modo live, cria Future associado ao contract_id para
        # receber resultado via _handle_contract_update (proposal_open_contract).
        if not self.dry_run and contract_id:
            loop = asyncio.get_running_loop()
            fut: asyncio.Future = loop.create_future()
            self._contract_futures[str(contract_id)] = fut

        # FIX B9: Mantém referência forte à task em _pending_tasks.
        task = asyncio.create_task(
            self._await_result(
                trade,
                proposal.get("payout", ask_price * 1.95),
                contract_id=str(contract_id),
            )
        )
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def _await_result(
        self,
        trade: Trade,
        expected_payout: float,
        contract_id: str = "",
    ) -> None:
        """
        Aguarda resultado do contrato.
        Em dry-run: simula resultado com probabilidade baseada na confiança.
        Em live: aguarda evento `proposal_open_contract` via Future registrado.
        """
        if self.dry_run:
            # Simula latência do contrato.
            await asyncio.sleep(5)
            won = random.random() < trade.confidence
            pnl = (expected_payout - trade.stake) if won else -trade.stake
            status = TradeStatus.WON if won else TradeStatus.LOST
            exit_price = trade.entry_price  # Simulado.
        else:
            # FIX C1: Aguarda resultado real via Future resolvido pelo listener
            # de proposal_open_contract. Timeout de 120s evita espera infinita.
            fut = self._contract_futures.get(contract_id)
            if fut is None:
                logger.error(
                    "Future não encontrado para contrato live.",
                    contract_id=contract_id,
                    trade_id=trade.id,
                )
                return

            try:
                contract_data = await asyncio.wait_for(fut, timeout=120.0)
                is_sold = contract_data.get("is_sold", 0)
                profit = float(contract_data.get("profit", 0.0))
                won = profit > 0
                pnl = profit
                status = TradeStatus.WON if won else TradeStatus.LOST
                exit_price = float(contract_data.get("exit_tick", trade.entry_price) or trade.entry_price)
            except asyncio.TimeoutError:
                logger.error(
                    "Timeout aguardando resultado do contrato live.",
                    contract_id=contract_id,
                    trade_id=trade.id,
                )
                status = TradeStatus.ERROR
                pnl = 0.0
                exit_price = None
                won = False
            finally:
                self._contract_futures.pop(contract_id, None)

        closed_at = datetime.now(tz=timezone.utc)

        # Atualiza banco.
        async with get_session() as db:
            repo = TradeRepository(db)
            await repo.update_result(
                trade_id=trade.id,
                status=status,
                exit_price=exit_price,
                pnl=pnl,
                payout=expected_payout if won else None,
                closed_at=closed_at,
            )

        # Atualiza estado da sessão.
        self._session_state.current_balance += pnl
        self._session_state.total_trades += 1
        if pnl > 0:
            self._session_state.wins += 1
            self._session_state.consecutive_losses = 0
        else:
            self._session_state.losses += 1
            self._session_state.consecutive_losses += 1

        total = self._session_state.total_trades
        self._session_state.win_rate = self._session_state.wins / total if total else 0.0

        # Registra no RiskManager e atualiza peak de saldo.
        trade.pnl = pnl
        trade.status = status
        trade.closed_at = closed_at
        self._risk_manager.register_trade(trade)
        # FIX B8: Atualiza high water mark para cálculo correto de drawdown.
        self._risk_manager.update_peak(self._session_state.current_balance)
        self._open_trades.pop(trade.id, None)

        logger.info(
            "Resultado do trade.",
            trade_id=trade.id,
            status=status.value,
            pnl=round(pnl, 4),
            balance=round(self._session_state.current_balance, 4),
            win_rate=round(self._session_state.win_rate, 4),
        )

    # ── Listener de Contratos Live ────────────────────────────────────────────

    async def _handle_contract_update(self, data: dict) -> None:
        """
        Callback para proposal_open_contract.
        Resolve o Future associado quando o contrato é encerrado.
        """
        contract = data.get("proposal_open_contract", {})
        contract_id = str(contract.get("contract_id", ""))
        is_sold = contract.get("is_sold", 0)

        if not is_sold:
            return  # Contrato ainda aberto — aguarda próximo evento.

        fut = self._contract_futures.get(contract_id)
        if fut and not fut.done():
            fut.set_result(contract)

    # ── Stop Events ───────────────────────────────────────────────────────────

    async def _record_stop_event(self, event_type: str, reason: str) -> None:
        try:
            async with get_session() as db:
                repo = StopEventRepository(db)
                await repo.record(
                    session_id=self.session_id,
                    event_type=event_type,
                    balance=self._session_state.current_balance,
                    pnl=self._session_state.current_balance
                        - self._session_state.initial_balance,
                    reason=reason,
                )
        except Exception as exc:
            logger.error("Falha ao registrar stop event.", error=str(exc))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_duration(self, strategy_name: str) -> int:
        durations = {
            "ema_rsi_macd": 5,
            "bollinger_reversion": 3,
            "breakout_squeeze": 5,
        }
        return durations.get(strategy_name, 5)

    def _get_duration_unit(self, strategy_name: str) -> str:
        # FIX B18: Implementado de verdade — cada estratégia pode ter unidade diferente.
        units = {
            "ema_rsi_macd": "t",        # Ticks.
            "bollinger_reversion": "t", # Ticks.
            "breakout_squeeze": "t",    # Ticks.
        }
        return units.get(strategy_name, "t")

    # ── Propriedades ──────────────────────────────────────────────────────────

    @property
    def session_state(self) -> SessionState:
        return self._session_state

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def metrics(self) -> dict:
        return self._risk_manager.session_metrics(self._session_state)