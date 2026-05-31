import asyncio
import numpy as np
from typing import Dict, Any
from PyQt6.QtCore import QThread, pyqtSignal, QObject
from src.core.deriv_ws import DerivWS
from src.core.consensus import ConsensusEngine
from src.core.risk_manager import RiskManager
from src.config import settings
from loguru import logger

class SignalEmitter(QObject):
    tick_update = pyqtSignal(object, str, float, object, float, float, int, float)
    log_message = pyqtSignal(str)
    connection_status = pyqtSignal(bool)

class OrchestratorThread(QThread):
    def __init__(self):
        super().__init__()
        # ✅ Passa config explicitamente para evitar NameError
        self.ws = DerivWS(app_id=settings.DERIV_APP_ID, token=settings.DERIV_TOKEN)
        self.consensus = ConsensusEngine()
        self.risk = RiskManager()
        self.emitter = SignalEmitter()
        self.prices = np.array([], dtype=float)
        self.state: Dict[str, Any] = {"balance": 1000.0, "drawdown": 0.0, "regime": "neutral"}
        self.trades_count = 0
        self.wins = 0
        self.running = False

    def run(self):
        self.running = True
        asyncio.run(self._async_loop())

    def stop(self):
        self.running = False
        self.ws.connected = False

    async def _async_loop(self):
        try:
            await self.ws.connect()
            self.emitter.connection_status.emit(True)
            self.emitter.log_message.emit("✅ Deriv WebSocket conectado")

            self.ws.on("tick", self._on_tick)
            self.ws.on("balance", self._on_balance)
            self.ws.on("trade_result", self._on_buy)

            # ✅ Config passada explicitamente
            await self.ws.subscribe_ticks(settings.SYMBOL)
            await self.ws.subscribe_balance()

            while self.running:
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            self.emitter.log_message.emit("🔴 Loop async finalizado")
        except Exception as e:
            self.emitter.log_message.emit(f"❌ Erro crítico: {e}")
        finally:
            await self.ws.close()

    async def _on_tick(self, data):
        quote = float(data["tick"]["quote"])
        self.prices = np.append(self.prices, quote)[-500:]
        
        if len(self.prices) >= 20:
            await self._execute_cycle()
        else:
            if len(self.prices) % 5 == 0:
                self.emitter.log_message.emit(f"📊 Coletando dados... {len(self.prices)}/20 ticks")

    async def _on_balance(self, data):
        bal = float(data["balance"]["balance"])
        self.state.update(self.risk.update(bal))
        self._update_ui_state()

    async def _on_buy(self, data):
        self.trades_count += 1
        self.emitter.log_message.emit(f"📈 Trade executado: ID {data.get('buy', {}).get('id', 'N/A')}")

    async def _execute_cycle(self):
        if self.risk.circuit_breaker:
            self.emitter.log_message.emit("🛑 Circuit Breaker ATIVO")
            return

        sig, score, votes = self.consensus.compute(self.prices, self.state)
        self._update_ui_state(sig, score, votes)

        if sig in ("CALL", "PUT") and not self.risk.circuit_breaker:
            stake = self.risk.get_stake()
            # ✅ Config passada explicitamente para execução
            await self.ws.execute_trade(sig, stake, settings.SYMBOL, settings.DURATION)
            self.emitter.log_message.emit(f"📤 {sig} | Stake: ${stake:.2f} | Conf: {score:.2%}")

    def _update_ui_state(self, signal="HOLD", score=0.0, votes=None):
        if votes is None: votes = {}
        pnl = self.risk.current_balance - self.risk.start_balance
        wr = (self.wins / self.trades_count * 100) if self.trades_count else 0.0
        self.emitter.tick_update.emit(
            self.prices.tolist(), signal, score, votes,
            self.state["drawdown"], pnl, self.trades_count, wr
        )