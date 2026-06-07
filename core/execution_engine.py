from __future__ import annotations

import asyncio
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

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
from ml.gemini_advisor import GeminiAdvisor, build_context_from_df
from ml.council.grand_oracle import GrandOracle, CouncilDecision
from ml.signal_quality_gate import get_signal_gate, SignalQualityGate


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
    # Integração com Gemini Advisor (opcional — None desativa o consultor)
    gemini_advisor: GeminiAdvisor | None = None
    # Callback assíncrono para broadcast de eventos (usado pelo Cloud API WebSocket)
    broadcast_fn: Callable[[str, Any], Awaitable[None]] | None = None
    # Oracle Council (8 agentes especializados + Gemini como 9º voto)
    grand_oracle: GrandOracle | None = None

    _strategies: list[StrategyBase] = field(default_factory=list, init=False)
    _risk_manager: RiskManager = field(init=False)
    _feature_engineer: FeatureEngineer = field(init=False)
    _session_state: SessionState = field(init=False)
    _open_trades: dict[str, Trade] = field(default_factory=dict, init=False)
    _symbol_locks: dict[str, asyncio.Lock] = field(
        default_factory=dict, init=False
    )
    _last_processed_epoch: dict[str, int] = field(
        default_factory=dict, init=False
    )
    # FIX B9: Mantém referência forte às tasks para evitar GC prematuro.
    _pending_tasks: set[asyncio.Task] = field(default_factory=set, init=False)
    # FIX C1: Mapa de contract_id → Future para receber resultado live.
    _contract_futures: dict[str, asyncio.Future] = field(default_factory=dict, init=False)
    _running: bool = field(default=False, init=False)
    # Rastreia resultados recentes por estratégia para o Gemini
    _strategy_results: dict[str, list[bool]] = field(default_factory=lambda: defaultdict(list), init=False)
    # Estratégia recomendada pelo Gemini (None = sem preferência)
    _gemini_priority: str | None = field(default=None, init=False)
    _gemini_confidence_mult: float = field(default=1.0, init=False)

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
        self._gemini_risk_flag: bool = False
        self._symbol_locks = {}
        self._last_processed_epoch = {}
        # Inicializa o Grand Oracle
        if self.grand_oracle is None:
            self.grand_oracle = GrandOracle()
            
        self._symbol_profiles: dict = {}   # symbol → SymbolProfile

    def set_symbol_profiles(self, profiles: dict) -> None:
        """Recebe perfis do HistoricalLoader após o boot."""
        self._symbol_profiles = profiles
        for sym, p in profiles.items():
            logger.info(
                "Perfil de símbolo aplicado.",
                symbol=sym,
                granularity=f"{p.granularity}s",
                duration=f"{p.duration}{p.duration_unit}",
                win_rate=f"{p.win_rate:.1%}",
            )

    def _get_symbol_lock(self, symbol: str) -> asyncio.Lock:
        lock = self._symbol_locks.get(symbol)
        if lock is None:
            lock = asyncio.Lock()
            self._symbol_locks[symbol] = lock
        return lock

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

        # Removemos tick listeners para evitar duplicação de pipeline
        # e confiamos apenas no polling baseado em candles.

        # FIX C1: Registra listener para resultados reais de contratos (modo live).
        if not self.dry_run:
            self.client.on("proposal_open_contract", self._handle_contract_update)
            task = asyncio.create_task(
                self.client.subscribe_open_contracts(),
                name="subscribe_open_contracts",
            )
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)

        self._running = True

        poll_task = asyncio.create_task(self._polling_loop(), name="engine_polling")
        self._pending_tasks.add(poll_task)
        poll_task.add_done_callback(self._pending_tasks.discard)

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

    async def _polling_loop(self) -> None:
        """
        Loop de polling: baixa velas atualizadas da Deriv e roda as estratégias.

        Executa a cada POLL_INTERVAL segundos, independente de subscriptions de
        ticks WebSocket. Garante que o bot funcione mesmo que a subscription
        de ticks falhe.
        """
        POLL_INTERVAL = 10  # segundos entre cada ciclo de avaliação
        COUNT = 500         # quantas velas buscar por ciclo (mantém histório longo)

        logger.info("Polling loop iniciado.", interval_s=POLL_INTERVAL)

        # Aguarda 2s para dar tempo ao startup — depois inicia imediatamente.
        await asyncio.sleep(2)

        while self._running:
            for symbol in list(self.symbol_manager.ready_symbols):
                try:
                    # Baixa as velas mais recentes da API Deriv
                    candles = await self.client.get_candles(
                        symbol=symbol,
                        granularity=self.symbol_manager._granularity,
                        count=COUNT,
                    )
                    if candles:
                        # Atualiza o DataFrame do SymbolManager com dados frescos
                        new_df = self.symbol_manager._candles_to_df(candles)
                        async with self.symbol_manager._lock:
                            self.symbol_manager._states[symbol].candles_df = new_df
                            self.symbol_manager._states[symbol].is_ready = True

                        logger.debug(
                            "Candles atualizados via polling.",
                            symbol=symbol,
                            count=len(candles),
                        )

                        # Roda o pipeline completo de estratégias com dados frescos
                        await self._process_symbol(symbol)

                except asyncio.CancelledError:
                    logger.info("Polling loop cancelado.")
                    return
                except Exception as exc:
                    logger.warning(
                        "Polling falhou para símbolo.",
                        symbol=symbol,
                        error=str(exc),
                    )

            await asyncio.sleep(POLL_INTERVAL)

        logger.info("Polling loop encerrado.")

    async def _process_symbol(self, symbol: str) -> None:
        """Pipeline completo para um símbolo em um tick/ciclo de polling."""
        
        lock = self._get_symbol_lock(symbol)
        async with lock:
            # 0. Ignora se já houver um trade aberto para este símbolo
            if any(t.symbol == symbol for t in self._open_trades.values()):
                return

            # 1. Candles + features.
            raw_df = self.symbol_manager.get_candles_df(symbol)
            if raw_df.empty or len(raw_df) < 55:
                return

            last_epoch = int(raw_df.iloc[-1]["epoch"])
            prev_epoch = self._last_processed_epoch.get(symbol)
            if prev_epoch == last_epoch:
                # Já processamos este candle mais recente; evita duplicidade
                return
            self._last_processed_epoch[symbol] = last_epoch

            cache = FeatureCache.get_instance()
            cached_df = cache.get_features(symbol, "features_df")

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

            # 2. Consulta Gemini Advisor (se disponível e for hora de consultar).
            await self._consult_gemini(feat_df, symbol)

            # 3. Ordena estratégias: Gemini prioriza uma delas movendo-a para frente.
            strategies = self._get_ordered_strategies()

            # 4. Avalia cada estratégia com filtro de qualidade
            quality_gate: SignalQualityGate = get_signal_gate()

            for strategy in strategies:
                signal = strategy.generate_signal(feat_df, symbol, self._session_state)
                if signal is None:
                    continue

                # ── SignalQualityGate: filtra falsos sinais ───────────────────
                quality_report = quality_gate.evaluate(signal, feat_df, self._session_state)
                if not quality_report.passed:
                    logger.debug(
                        "Sinal rejeitado pelo QualityGate.",
                        symbol=symbol,
                        reason=quality_report.rejection_reason,
                    )
                    continue

                # Aplica boost de qualidade à confiança
                if quality_report.boost != 1.0:
                    signal = Signal(
                        symbol=signal.symbol,
                        direction=signal.direction,
                        confidence=min(1.0, signal.confidence * quality_report.boost),
                        strategy_name=signal.strategy_name,
                        model_name=signal.model_name,
                        contract_type=signal.contract_type,
                        entry_price=signal.entry_price,
                    )

                # Aplica o multiplicador de confiança do Gemini
                if self._gemini_confidence_mult != 1.0:
                    signal = Signal(
                        symbol=signal.symbol,
                        direction=signal.direction,
                        confidence=min(1.0, signal.confidence * self._gemini_confidence_mult),
                        strategy_name=signal.strategy_name,
                        model_name=signal.model_name,
                        contract_type=signal.contract_type,
                        entry_price=signal.entry_price,
                    )

                # ── Oracle Council: votação dos especialistas ────────────────
                if self.grand_oracle is not None:
                    peer_dfs = {
                        s: self._feature_engineer.compute(
                            self.symbol_manager.get_candles_df(s)
                        )
                        for s in self.symbol_manager.ready_symbols
                        if s != symbol
                    }
                    peer_dfs = {s: df for s, df in peer_dfs.items() if not df.empty}

                    ticks = self.symbol_manager.get_recent_ticks(symbol)
                    decision = self.grand_oracle.decide(
                        signal=signal,
                        df=feat_df,
                        session=self._session_state,
                        ticks=ticks,
                        peer_dfs=peer_dfs or None,
                    )

                    if self.broadcast_fn:
                        votes_list = []
                        for v in decision.votes.values():
                            votes_list.append({
                                "agent": v.agent_name,
                                "action": v.action,
                                "confidence": round(v.score, 4),
                                "veto": v.veto,
                                "reasoning": v.reasoning,
                            })
                        summary = {
                            "approved": decision.approved,
                            "action": decision.direction,
                            "confidence": decision.confidence,
                            "veto_by": decision.vetoed_by,
                            "reasoning": decision.reasoning,
                            "votes": votes_list,
                        }
                        await self.broadcast_fn("council_vote", summary)

                    if not decision.approved:
                        logger.info(
                            "Oracle Council bloqueou o trade.",
                            symbol=symbol,
                            score=decision.confidence,
                            veto_by=decision.vetoed_by,
                        )
                        break

                    signal = Signal(
                        symbol=signal.symbol,
                        direction=signal.direction,
                        confidence=min(1.0, decision.confidence),
                        strategy_name=signal.strategy_name,
                        model_name=f"council({decision.confidence:.2f})",
                        contract_type=signal.contract_type,
                        entry_price=signal.entry_price,
                    )

                await self._handle_signal(signal)
                break

    async def _consult_gemini(self, feat_df, symbol: str) -> None:
        """Consulta o Gemini Advisor e atualiza as prioridades/flags."""
        if self.gemini_advisor is None or not self.gemini_advisor.should_consult():
            return
        strategy_names = [s.name for s in self._strategies]
        recent_results = {
            name: (
                sum(self._strategy_results[name][-20:]) / len(self._strategy_results[name][-20:])
                if self._strategy_results[name] else 0.5
            )
            for name in strategy_names
        }
        ctx = build_context_from_df(
            df=feat_df,
            symbol=symbol,
            session_state=self._session_state,
            available_strategies=strategy_names,
            strategy_results=recent_results,
        )
        advice = await self.gemini_advisor.consult(ctx)
        if advice:
            self._gemini_priority = advice.recommended_strategy
            self._gemini_confidence_mult = advice.confidence_multiplier
            self._gemini_risk_flag = advice.risk_flag
            if self.broadcast_fn:
                await self.broadcast_fn("gemini_advice", {
                    "strategy": advice.recommended_strategy,
                    "multiplier": advice.confidence_multiplier,
                    "risk_flag": advice.risk_flag,
                    "reasoning": advice.reasoning,
                })

    def _get_ordered_strategies(self) -> list[StrategyBase]:
        """Retorna estratégias ordenadas com a prioridade do Gemini na frente."""
        if not self._gemini_priority:
            return self._strategies
        prioritized = [s for s in self._strategies if s.name == self._gemini_priority]
        rest = [s for s in self._strategies if s.name != self._gemini_priority]
        return prioritized + rest

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
                duration=self._get_duration(signal.strategy_name, signal.symbol),
                duration_unit=self._get_duration_unit(signal.strategy_name, signal.symbol),
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
            if self.broadcast_fn:
                await self.broadcast_fn("trade_opened", {
                    "trade_id": trade.id,
                    "symbol": trade.symbol,
                    "direction": trade.direction.value,
                    "stake": ask_price,
                    "strategy": trade.strategy_name,
                    "confidence": round(trade.confidence, 4),
                    "status": "OPEN",
                    "ts": trade.opened_at.isoformat(),
                })
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
        try:
            if self.dry_run:
                await asyncio.sleep(5)
                won = random.random() < trade.confidence
                pnl = (expected_payout - trade.stake) if won else -trade.stake
                status = TradeStatus.WON if won else TradeStatus.LOST
                exit_price = trade.entry_price
            else:
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
                except asyncio.TimeoutError:
                    logger.error(
                        "Timeout aguardando resultado do contrato live.",
                        contract_id=contract_id,
                        trade_id=trade.id,
                    )
                    status = TradeStatus.ERROR
                    pnl = 0.0
                    won = False
                    exit_price = trade.entry_price
                else:
                    profit = float(contract_data.get("profit", 0.0))
                    won = profit > 0
                    pnl = profit
                    status = TradeStatus.WON if won else TradeStatus.LOST
                    raw_exit = contract_data.get("exit_tick")
                    exit_price = float(raw_exit) if raw_exit is not None else (trade.entry_price or 0.0)

            closed_at = datetime.now(tz=timezone.utc)

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

            trade.pnl = pnl
            trade.status = status
            trade.closed_at = closed_at
            self._risk_manager.register_trade(trade)
            self._risk_manager.update_peak(self._session_state.current_balance)

            self._strategy_results[trade.strategy_name].append(won)
            if len(self._strategy_results[trade.strategy_name]) > 50:
                self._strategy_results[trade.strategy_name].pop(0)

            # Notifica QualityGate e Oracle do resultado
            quality_gate = get_signal_gate()
            quality_gate.record_outcome(trade, won)
            if self.grand_oracle is not None:
                self.grand_oracle.notify_outcome(
                    action=trade.direction.value if hasattr(trade.direction, "value") else str(trade.direction),
                    won=won,
                    pnl=pnl,
                    signal=getattr(trade, "signal_name", trade.strategy_name),
                )

            logger.info(
                "Resultado do trade.",
                trade_id=trade.id,
                status=status.value,
                pnl=round(pnl, 4),
                balance=round(self._session_state.current_balance, 4),
                win_rate=round(self._session_state.win_rate, 4),
            )

            if self.broadcast_fn:
                await self.broadcast_fn("trade", {
                    "trade_id": trade.id,
                    "symbol": trade.symbol,
                    "direction": trade.direction.value,
                    "stake": trade.stake,
                    "strategy": trade.strategy_name,
                    "confidence": round(trade.confidence, 4),
                    "pnl": round(pnl, 4),
                    "status": status.value,
                    "balance": round(self._session_state.current_balance, 4),
                    "win_rate": round(self._session_state.win_rate, 4),
                    "ts": closed_at.isoformat(),
                })

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "Erro inesperado em _await_result.",
                error=str(exc),
                trade_id=trade.id,
                contract_id=contract_id,
            )
        finally:
            if contract_id:
                self._contract_futures.pop(contract_id, None)
            self._open_trades.pop(trade.id, None)

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

    def _get_duration(self, strategy_name: str, symbol: str = "") -> int:
        profile = self._symbol_profiles.get(symbol)
        if profile:
            return profile.duration
        # fallback por estratégia
        return {"ema_rsi_macd": 5, "bollinger_reversion": 3, "breakout_squeeze": 5}.get(strategy_name, 5)

    def _get_duration_unit(self, strategy_name: str, symbol: str = "") -> str:
        profile = self._symbol_profiles.get(symbol)
        if profile:
            return profile.duration_unit
        return "t"

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