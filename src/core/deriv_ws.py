import asyncio
import json
import websockets
from typing import Callable, Dict, Any
from loguru import logger
from src.config import settings


class DerivWS:
    def __init__(self, app_id: int = 1089, token: str = ""):
        self.app_id = app_id
        self.token = token
        self.ws = None
        self.req_id = 0
        self.callbacks: Dict[str, list[Callable]] = {}
        self.connected = False
        self._lock = asyncio.Lock()
        self._listen_task = None

    def on(self, event: str, callback: Callable):
        self.callbacks.setdefault(event, []).append(callback)

    async def connect(self):
        url = f"wss://ws.derivws.com/websockets/v3?app_id={self.app_id}"
        try:
            self.ws = await websockets.connect(url, ping_interval=20, ping_timeout=15)
            self.connected = True
            logger.info(f"🔑 Autenticando com token: {self.token[:4]}...{self.token[-4:]}")
            await self._send({"authorize": self.token})

            resp = await self._wait_for_response("authorize", timeout=5.0)
            if "error" in resp:
                raise ConnectionError(f"❌ Deriv rejeitou o token: {resp['error']['message']}")
            logger.info("✅ Autenticação Deriv bem-sucedida")

            self._listen_task = asyncio.create_task(self._listen())
        except Exception as e:
            logger.error(f"❌ Falha na conexão: {e}")
            self.connected = False
            raise

    async def _wait_for_response(self, expected_type: str, timeout: float = 5.0):
        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < timeout:
            try:
                raw = await asyncio.wait_for(self.ws.recv(), timeout=0.5)
                msg = json.loads(raw)
                if msg.get("msg_type") == expected_type or "error" in msg:
                    return msg
            except asyncio.TimeoutError:
                continue
        raise TimeoutError(f"Timeout aguardando resposta '{expected_type}'")

    async def _listen(self):
        try:
            async for raw in self.ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if "error" in msg:
                    logger.error(f"🚨 Deriv Error: {msg['error']['message']}")
                    continue

                if "tick" in msg:
                    await self._dispatch("tick", msg)
                elif "balance" in msg:
                    await self._dispatch("balance", msg)
                elif "buy" in msg or "proposal_open_contract" in msg:
                    await self._dispatch("trade_result", msg)
                elif "ping" in msg:
                    await self._send({"pong": 1})
        except websockets.ConnectionClosed:
            logger.warning("⚠️ WebSocket desconectado")
            self.connected = False

    async def _dispatch(self, event: str, data: Any):
        if event in self.callbacks:
            for cb in self.callbacks[event]:
                try:
                    if asyncio.iscoroutinefunction(cb):
                        await cb(data)
                    else:
                        cb(data)
                except Exception as e:
                    logger.error(f"❌ Erro no callback '{event}': {e}")

    async def _send(self, payload: dict):
        async with self._lock:
            self.req_id += 1
            payload["req_id"] = self.req_id
            await self.ws.send(json.dumps(payload))
        return self.req_id

    def subscribe_ticks(self, symbol: str):
        logger.info(f"📡 Subscribing to ticks for {symbol}...")
        return self._send({"ticks": symbol, "subscribe": 1})

    def subscribe_balance(self):
        logger.info("📡 Subscribing to balance updates...")
        return self._send({"balance": 1, "subscribe": 1})

    def execute_trade(self, direction: str, stake: float, symbol: str, duration: int):
        logger.info(f"📤 Executando trade: {direction} | Stake: ${stake:.2f} | Duração: {duration}{settings.DURATION_UNIT}")
        return self._send({
            "buy": 1,
            "price": stake,
            "parameters": {
                "amount": str(stake), 
                "basis": "stake", 
                "contract_type": direction,
                "currency": "USD", 
                "duration": duration, 
                "duration_unit": settings.DURATION_UNIT,  # Dinâmico baseado no config.py
                "symbol": symbol
            }
        })

    async def close(self):
        self.connected = False
        if self._listen_task:
            self._listen_task.cancel()
            try: 
                await self._listen_task
            except asyncio.CancelledError: 
                pass
        if self.ws:
            await self.ws.close()