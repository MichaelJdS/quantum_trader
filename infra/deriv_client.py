from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import websockets
from loguru import logger
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from websockets.exceptions import ConnectionClosed, WebSocketException

from core.exceptions import DerivAuthorizationError, DerivConnectionError, DerivContractError
from core.settings import get_settings


# ── Tipos ─────────────────────────────────────────────────────────────────────

MessageCallback = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


class CircuitState(Enum):
    CLOSED = auto()    # Operação normal.
    OPEN = auto()      # Bloqueado — muitas falhas.
    HALF_OPEN = auto() # Testando recuperação.


# ── Circuit Breaker ───────────────────────────────────────────────────────────

@dataclass
class CircuitBreaker:
    """
    Circuit Breaker para o WebSocket Deriv.

    Estados:
      CLOSED   → operação normal, falhas são contadas.
      OPEN     → bloqueado por `reset_timeout` segundos após `threshold` falhas.
      HALF_OPEN→ uma tentativa de reconexão é permitida.
    """

    threshold: int = 5
    reset_timeout: float = 60.0
    _failures: int = field(default=0, init=False, repr=False)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False, repr=False)
    _opened_at: float = field(default=0.0, init=False, repr=False)

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.threshold:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            logger.error(
                "Circuit Breaker ABERTO — muitas falhas de conexão.",
                failures=self._failures,
                reset_in=self.reset_timeout,
            )

    def record_success(self) -> None:
        self._failures = 0
        self._state = CircuitState.CLOSED
        logger.success("Circuit Breaker FECHADO — conexão restabelecida.")

    def allow_request(self) -> bool:
        if self._state == CircuitState.CLOSED:
            return True
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self.reset_timeout:
                self._state = CircuitState.HALF_OPEN
                logger.info("Circuit Breaker HALF_OPEN — tentando reconexão.")
                return True
            return False
        # HALF_OPEN: permite uma tentativa.
        return True

    @property
    def state(self) -> CircuitState:
        return self._state


# ── Rate Limiter ──────────────────────────────────────────────────────────────

class RateLimiter:
    """
    Token bucket para respeitar o limite de 50 req/s da API Deriv.
    https://api.deriv.com/api-explorer#rate-limit
    """

    def __init__(self, rate: int = 50, period: float = 1.0) -> None:
        self._rate = rate
        self._period = period
        self._tokens = float(rate)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            refill = elapsed * (self._rate / self._period)
            self._tokens = min(self._rate, self._tokens + refill)
            self._last_refill = now

            if self._tokens < 1.0:
                wait_time = (1.0 - self._tokens) / (self._rate / self._period)
                await asyncio.sleep(wait_time)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


# ── Cliente Principal ─────────────────────────────────────────────────────────

class DerivClient:
    """
    Cliente WebSocket assíncrono para a API Deriv.

    Funcionalidades:
      - Autenticação com token.
      - Reconnect automático com backoff exponencial (Tenacity).
      - Circuit Breaker para falhas repetidas.
      - Rate limiting (50 req/s).
      - Request-response via Future (req_id → Future).
      - Sistema de callbacks por tipo de mensagem.
      - Subscriptions de ticks e candles com gestão de sub_id.
      - Modo dry-run para testes sem execução real.
    """

    def __init__(self, dry_run: bool = True) -> None:
        self._settings = get_settings()
        self._dry_run = dry_run
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._req_id: int = 0
        self._pending: dict[int, asyncio.Future[dict]] = {}
        self._callbacks: dict[str, list[MessageCallback]] = {}
        self._subscriptions: dict[str, str] = {}  # symbol → sub_id
        self._authorized: bool = False
        self._running: bool = False
        self._circuit: CircuitBreaker = CircuitBreaker(threshold=5, reset_timeout=60.0)
        self._rate_limiter: RateLimiter = RateLimiter(rate=40)  # 40/s: margem segura
        self._listen_task: asyncio.Task | None = None
        self._ping_task: asyncio.Task | None = None

    # ── Conexão ───────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """
        Conecta ao WebSocket Deriv com retry exponencial e circuit breaker.
        Reconecta automaticamente em caso de queda.
        """
        if not self._circuit.allow_request():
            raise DerivConnectionError(
                f"Circuit Breaker aberto. Aguarde {self._circuit.reset_timeout}s."
            )

        url = (
            f"{self._settings.deriv_websocket_url}"
            f"?app_id={self._settings.deriv_app_id}"
        )

        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type((WebSocketException, OSError, ConnectionError)),
                wait=wait_exponential(multiplier=1, min=2, max=60),
                stop=stop_after_attempt(self._settings.max_retries
                                        if hasattr(self._settings, "max_retries") else 10),
                reraise=True,
            ):
                with attempt:
                    logger.info(
                        "Conectando ao Deriv WebSocket...",
                        url=url.split("?")[0],
                        attempt=attempt.retry_state.attempt_number,
                    )
                    self._ws = await websockets.connect(
                        url,
                        ping_interval=30,
                        ping_timeout=10,
                        close_timeout=5,
                        max_size=2**20,  # 1MB
                    )
                    # _listen_loop deve iniciar ANTES de _authorize()
                    # para que os Futures em _pending sejam resolvidos.
                    self._running = True
                    self._listen_task = asyncio.create_task(
                        self._listen_loop(), name="deriv_listen"
                    )
                    # Dá um yield ao loop para que a task seja agendada.
                    await asyncio.sleep(0)
                    await self._authorize()
                    self._circuit.record_success()
                    self._ping_task = asyncio.create_task(
                        self._ping_loop(), name="deriv_ping"
                    )
                    logger.success(
                        "Conectado e autorizado.",
                        mode="DRY-RUN" if self._dry_run else "LIVE",
                    )
                    return

        except Exception as exc:
            self._circuit.record_failure()
            raise DerivConnectionError(f"Falha na conexão: {exc}") from exc

    async def disconnect(self) -> None:
        """Encerra a conexão de forma limpa."""
        self._running = False

        if self._listen_task:
            self._listen_task.cancel()
        if self._ping_task:
            self._ping_task.cancel()

        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()

        if self._ws and not self._ws.closed:
            await self._ws.close()

        logger.info("DerivClient desconectado.")

    # ── Autorização ───────────────────────────────────────────────────────────

    async def _authorize(self) -> None:
        response = await self._send_raw(
            {"authorize": self._settings.deriv_api_token},
            timeout=15.0,
        )
        if "error" in response:
            raise DerivAuthorizationError(
                response["error"].get("message", "Autorização falhou.")
            )
        self._authorized = True
        account = response.get("authorize", {})
        logger.info(
            "Autorizado.",
            email=account.get("email", "?"),
            balance=account.get("balance"),
            currency=account.get("currency"),
        )

    # ── Loop de Escuta ────────────────────────────────────────────────────────

    async def _listen_loop(self) -> None:
        """Processa mensagens recebidas do WebSocket em loop contínuo."""
        while self._running:
            # FIX B4: Verifica se _ws ainda existe antes de chamar recv().
            # Durante reconexão, _ws pode ser None temporariamente.
            if self._ws is None or self._ws.closed:
                logger.warning("_listen_loop: WebSocket indisponível — aguardando reconexão.")
                await asyncio.sleep(0.5)
                continue

            try:
                raw = await self._ws.recv()
                data: dict = json.loads(raw)

                # Resolve futures de req/resp.
                req_id = data.get("req_id")
                if req_id and req_id in self._pending:
                    fut = self._pending.pop(req_id)
                    if not fut.done():
                        fut.set_result(data)

                # Despacha callbacks por tipo de mensagem.
                msg_type = data.get("msg_type")
                if msg_type and msg_type in self._callbacks:
                    for cb in self._callbacks[msg_type]:
                        asyncio.create_task(cb(data))

            except ConnectionClosed as exc:
                logger.warning("Conexão fechada.", code=exc.code, reason=exc.reason)
                if self._running:
                    asyncio.create_task(self._reconnect())
                break

            except json.JSONDecodeError as exc:
                logger.error("Mensagem inválida do WebSocket.", error=str(exc))

            except Exception as exc:
                logger.exception("Erro inesperado no listen_loop.", error=str(exc))
                break

    async def _ping_loop(self) -> None:
        """Envia ping periódico para manter a conexão ativa."""
        while self._running:
            await asyncio.sleep(25)
            try:
                await self._send_raw({"ping": 1}, timeout=5.0)
            except Exception:
                logger.warning("Ping falhou — reconectando.")
                await self._reconnect()
                break

    async def _reconnect(self) -> None:
        """Tenta reconectar após queda."""
        logger.info("Iniciando reconexão...")
        self._authorized = False
        await asyncio.sleep(2)
        try:
            await self.connect()
            # Re-inscreve em todos os símbolos ativos.
            for symbol in list(self._subscriptions.keys()):
                self._subscriptions.pop(symbol, None)
                await self.subscribe_ticks(symbol)
        except DerivConnectionError as exc:
            logger.error("Reconexão falhou.", error=str(exc))

    # ── Envio de Mensagens ────────────────────────────────────────────────────

    async def _send_raw(
        self,
        payload: dict,
        timeout: float = 10.0,
    ) -> dict:
        """
        Envia mensagem e aguarda resposta via Future (req_id matching).
        Aplica rate limiting antes do envio.
        """
        await self._rate_limiter.acquire()

        self._req_id += 1
        payload["req_id"] = self._req_id

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict] = loop.create_future()
        self._pending[self._req_id] = fut

        try:
            await self._ws.send(json.dumps(payload))
        except Exception as exc:
            self._pending.pop(self._req_id, None)
            raise DerivConnectionError(f"Falha ao enviar mensagem: {exc}") from exc

        try:
            return await asyncio.wait_for(asyncio.shield(fut), timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(self._req_id, None)
            raise TimeoutError(f"Timeout ({timeout}s) aguardando resposta. payload={payload}")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def on(self, msg_type: str, callback: MessageCallback) -> None:
        """Registra callback assíncrono para tipo de mensagem."""
        self._callbacks.setdefault(msg_type, []).append(callback)

    def off(self, msg_type: str, callback: MessageCallback) -> None:
        """Remove callback registrado."""
        if msg_type in self._callbacks:
            self._callbacks[msg_type] = [
                cb for cb in self._callbacks[msg_type] if cb is not callback
            ]

    # ── Subscriptions ─────────────────────────────────────────────────────────

    async def subscribe_ticks(
        self,
        symbol: str,
        callback: MessageCallback | None = None,
    ) -> str:
        """
        Inscreve em stream de ticks de um símbolo.

        Returns:
            sub_id da subscription para eventual cancelamento.
        """
        if symbol in self._subscriptions:
            logger.debug("Já inscrito.", symbol=symbol)
            return self._subscriptions[symbol]

        if callback:
            self.on("tick", callback)

        response = await self._send_raw(
            {"ticks": symbol, "subscribe": 1},
            timeout=10.0,
        )
        if "error" in response:
            raise DerivConnectionError(
                f"Erro ao inscrever ticks {symbol}: {response['error']['message']}"
            )

        sub_id = response.get("subscription", {}).get("id", "")
        self._subscriptions[symbol] = sub_id
        logger.info("Inscrito em ticks.", symbol=symbol, sub_id=sub_id)
        return sub_id

    async def unsubscribe_ticks(self, symbol: str) -> None:
        """Cancela subscription de ticks de um símbolo."""
        sub_id = self._subscriptions.pop(symbol, None)
        if not sub_id:
            return
        await self._send_raw({"forget": sub_id}, timeout=5.0)
        logger.info("Desinscrito de ticks.", symbol=symbol)

    async def unsubscribe_all(self) -> None:
        """Cancela todas as subscriptions ativas."""
        await self._send_raw({"forget_all": "ticks"}, timeout=5.0)
        self._subscriptions.clear()
        logger.info("Todas as subscriptions canceladas.")

    # ── Dados de Mercado ──────────────────────────────────────────────────────

    async def get_candles(
        self,
        symbol: str,
        granularity: int = 60,
        count: int = 500,
    ) -> list[dict]:
        """
        Obtém candles OHLCV históricos.

        Args:
            symbol: Ex: "R_50", "R_75".
            granularity: Segundos por candle (60, 300, 3600...).
            count: Quantidade de candles (max 5000).

        Returns:
            Lista de dicts com open, high, low, close, epoch.
        """
        response = await self._send_raw(
            {
                "ticks_history": symbol,
                "style": "candles",
                "granularity": granularity,
                "count": min(count, 5000),
                "end": "latest",
            },
            timeout=20.0,
        )
        if "error" in response:
            raise DerivConnectionError(
                f"Erro ao buscar candles {symbol}: {response['error']['message']}"
            )
        return response.get("candles", [])

    async def get_tick_history(
        self,
        symbol: str,
        count: int = 500,
    ) -> list[dict]:
        """Obtém histórico de ticks recentes."""
        response = await self._send_raw(
            {
                "ticks_history": symbol,
                "style": "ticks",
                "count": min(count, 5000),
                "end": "latest",
            },
            timeout=20.0,
        )
        if "error" in response:
            raise DerivConnectionError(
                f"Erro ao buscar ticks {symbol}: {response['error']['message']}"
            )
        history = response.get("history", {})
        prices = history.get("prices", [])
        times = history.get("times", [])
        return [{"price": p, "epoch": t} for p, t in zip(prices, times)]

    async def get_active_symbols(self) -> list[dict]:
        """Retorna lista de símbolos ativos na plataforma."""
        response = await self._send_raw(
            {"active_symbols": "brief", "product_type": "basic"},
            timeout=15.0,
        )
        return response.get("active_symbols", [])

    async def get_balance(self) -> dict:
        """Retorna saldo atual da conta."""
        response = await self._send_raw({"balance": 1}, timeout=10.0)
        if "error" in response:
            raise DerivConnectionError(response["error"]["message"])
        return response.get("balance", {})

    # ── Propostas e Contratos ─────────────────────────────────────────────────

    async def get_proposal(
        self,
        symbol: str,
        contract_type: str,
        duration: int,
        duration_unit: str,
        amount: float,
        basis: str = "stake",
        currency: str = "USD",
    ) -> dict:
        """
        Obtém cotação de contrato antes de comprar.

        Args:
            symbol: Símbolo (ex: "R_50").
            contract_type: "CALL" ou "PUT".
            duration: Duração numérica.
            duration_unit: "t" (ticks), "s" (segundos), "m" (minutos).
            amount: Valor do stake.
            basis: "stake" ou "payout".
            currency: Moeda da conta.
        """
        response = await self._send_raw(
            {
                "proposal": 1,
                "contract_type": contract_type,
                "symbol": symbol,
                "duration": duration,
                "duration_unit": duration_unit,
                "amount": amount,
                "basis": basis,
                "currency": currency,
            },
            timeout=15.0,
        )
        if "error" in response:
            raise DerivContractError(
                f"Proposta falhou [{symbol}/{contract_type}]: "
                f"{response['error']['message']}"
            )
        return response.get("proposal", {})

    async def buy_contract(
        self,
        proposal_id: str,
        price: float,
    ) -> dict:
        """
        Compra contrato a partir de um proposal_id.

        Em dry_run, simula a compra sem enviar para a API.
        """
        if self._dry_run:
            logger.warning(
                "DRY-RUN: compra simulada, nenhum contrato real executado.",
                proposal_id=proposal_id,
                price=price,
            )
            return {
                "buy": {
                    "contract_id": f"DRY_{proposal_id}",
                    "buy_price": price,
                    "transaction_id": 0,
                }
            }

        response = await self._send_raw(
            {"buy": proposal_id, "price": price},
            timeout=15.0,
        )
        if "error" in response:
            raise DerivContractError(
                f"Compra falhou: {response['error']['message']}"
            )
        return response

    async def get_open_contracts(self) -> list[dict]:
        """Retorna lista de contratos abertos."""
        response = await self._send_raw(
            {"portfolio": 1},
            timeout=10.0,
        )
        return response.get("portfolio", {}).get("contracts", [])

    async def sell_contract(self, contract_id: int, price: float = 0) -> dict:
        """Vende (fecha) contrato aberto pelo preço atual ou mínimo."""
        if self._dry_run:
            logger.warning("DRY-RUN: venda simulada.", contract_id=contract_id)
            return {"sell": {"sold_for": price}}

        response = await self._send_raw(
            {"sell": contract_id, "price": price},
            timeout=15.0,
        )
        if "error" in response:
            raise DerivContractError(f"Venda falhou: {response['error']['message']}")
        return response

    # ── Propriedades ──────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and not self._ws.closed and self._authorized

    @property
    def is_dry_run(self) -> bool:
        return self._dry_run

    @property
    def circuit_state(self) -> CircuitState:
        return self._circuit.state

    @property
    def active_subscriptions(self) -> list[str]:
        return list(self._subscriptions.keys())