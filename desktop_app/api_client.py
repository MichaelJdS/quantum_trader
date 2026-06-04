"""
desktop_app/api_client.py — Cliente WebSocket + REST para o backend Cloud

Conecta o App Windows ao backend no Google Cloud (Compute Engine).
Reconexão automática com backoff exponencial.
Usa thread dedicada com loop asyncio separado do Qt para evitar conflitos.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any, Callable
from urllib.parse import urlencode

import httpx
from loguru import logger


class CloudAPIClient:
    """
    Cliente assíncrono para a Cloud API do Quantum Trader.

    Suporta:
    - Chamadas REST (start, stop, status, gemini chat, config)
    - Stream de eventos via WebSocket (trades, logs, métricas ao vivo)
    - Reconexão automática com backoff exponencial
    - Thread dedicada com asyncio loop próprio (compatível com PyQt6)
    """

    def __init__(
        self,
        base_url: str,
        api_token: str = "",
        on_event: Callable[[str, Any], None] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        self.api_token = api_token
        self.on_event = on_event
        self._reconnect = False
        self._connected = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    # ── Stream WebSocket (thread dedicada) ───────────────────────────────────

    def start_stream(self) -> None:
        """Inicia stream WebSocket em thread dedicada com loop asyncio próprio."""
        if self._thread and self._thread.is_alive():
            return
        self._reconnect = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="qt-ws-client")
        self._thread.start()

    def stop_stream(self) -> None:
        """Para o stream e encerra o loop."""
        self._reconnect = False
        self._connected = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _run_loop(self) -> None:
        """Roda o event loop asyncio nesta thread dedicada."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._ws_loop())
        except Exception as exc:
            logger.warning("WebSocket loop encerrado.", error=str(exc))
        finally:
            self._loop.close()

    async def _ws_loop(self) -> None:
        """Loop de reconexão automática."""
        backoff = 1.0
        while self._reconnect:
            try:
                await self._ws_connect()
                backoff = 1.0
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._connected = False
                self._emit("connection_lost", {})
                logger.warning("WS desconectado. Reconectando...", error=str(exc), wait=backoff)
                if self._reconnect:
                    await asyncio.sleep(min(backoff, 60.0))
                    backoff = min(backoff * 2, 60.0)

    async def _ws_connect(self) -> None:
        """Conecta ao WebSocket e processa mensagens."""
        try:
            import websockets
        except ImportError:
            logger.error("websockets não instalado. Execute: pip install websockets")
            await asyncio.sleep(30)
            return

        params = urlencode({"token": self.api_token}) if self.api_token else ""
        url = f"{self.ws_url}/ws{'?' + params if params else ''}"
        logger.info("WS conectando...", url=url)

        async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
            self._connected = True
            self._emit("connected", {})
            logger.success("WebSocket conectado.")
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=35.0)
                    msg = json.loads(raw)
                    event = msg.get("event", "")
                    data = msg.get("data", {})
                    if event and event not in ("heartbeat",):
                        self._emit(event, data)
                    elif msg.get("type") == "heartbeat":
                        pass
                except TimeoutError:
                    await ws.send(json.dumps({"type": "ping"}))

    def _emit(self, event: str, data: Any) -> None:
        if self.on_event:
            try:
                self.on_event(event, data)
            except Exception as exc:
                logger.error("Erro no callback.", event=event, error=str(exc))

    # ── REST (chamadas síncronas via thread) ─────────────────────────────────

    def _call_sync(self, coro) -> Any:
        """Executa uma co-rotina de forma síncrona em thread separada."""
        return asyncio.run(coro)

    def start_bot_sync(self, config: dict) -> dict:
        return self._call_sync(self._post("/start", config))

    def stop_bot_sync(self) -> dict:
        return self._call_sync(self._post("/stop", {}))

    def get_status_sync(self) -> dict:
        return self._call_sync(self._get("/status"))

    def get_council_status_sync(self) -> dict:
        return self._call_sync(self._get("/council/status"))

    def chat_sync(self, message: str) -> str:
        resp = self._call_sync(self._post("/gemini/chat", {"message": message}))
        return resp.get("message", "")

    def update_config_sync(self, config: dict) -> dict:
        return self._call_sync(self._put("/config", config))

    def health_check_sync(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.base_url}/health")
                return resp.status_code == 200
        except Exception:
            return False

    # ── Versões assíncronas (para uso em co-rotinas) ─────────────────────────

    async def start_bot(self, config: dict) -> dict:
        return await self._post("/start", config)

    async def stop_bot(self) -> dict:
        return await self._post("/stop", {})

    async def get_status(self) -> dict:
        return await self._get("/status")

    async def chat(self, message: str) -> str:
        resp = await self._post("/gemini/chat", {"message": message})
        return resp.get("message", "")

    async def update_config(self, config: dict) -> dict:
        return await self._put("/config", config)

    # ── HTTP helpers ─────────────────────────────────────────────────────────

    def _make_headers(self) -> dict:
        if self.api_token:
            return {"Authorization": f"Bearer {self.api_token}"}
        return {}

    async def _post(self, path: str, body: dict) -> dict:
        async with httpx.AsyncClient(headers=self._make_headers(), timeout=15.0) as client:
            try:
                resp = await client.post(f"{self.base_url}{path}", json=body)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(exc.response.json().get("detail", str(exc))) from exc
            except Exception as exc:
                raise RuntimeError(f"Erro de conexão: {exc}") from exc

    async def _get(self, path: str) -> dict:
        async with httpx.AsyncClient(headers=self._make_headers(), timeout=15.0) as client:
            try:
                resp = await client.get(f"{self.base_url}{path}")
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(exc.response.json().get("detail", str(exc))) from exc
            except Exception as exc:
                raise RuntimeError(f"Erro de conexão: {exc}") from exc

    async def _put(self, path: str, body: dict) -> dict:
        async with httpx.AsyncClient(headers=self._make_headers(), timeout=15.0) as client:
            try:
                resp = await client.put(f"{self.base_url}{path}", json=body)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(exc.response.json().get("detail", str(exc))) from exc
            except Exception as exc:
                raise RuntimeError(f"Erro de conexão: {exc}") from exc

    def close(self) -> None:
        self.stop_stream()
