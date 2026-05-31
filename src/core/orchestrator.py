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
        self.ws = DerivWS(app_id=settings.DERIV_APP_ID, token=settings.DERIV_TOKEN)
        self.consensus = ConsensusEngine()
        self.risk = RiskManager()
        self.emitter = SignalEmitter()
        self.prices = np.array([], dtype=float)
        self.state: Dict[str, Any] = {"balance": 1000.0, "drawdown": 0.0, "regime": "neutral"}
        self.trades_count = 0
        self.wins = 0
        self.running = False
        self.in_trade = False

    def run(self):
        self.running = True
        # Cria um novo event loop para a thread do worker
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_loop())
        finally:
            loop.close()

    def stop(self):
        self.running = False
        self.ws.connected = False

    async def _async_loop(self):
        try:
            await self.ws.connect()
            self.emitter.connection_status.emit(True)
            self.emitter.log_message.emit("✅ Deriv WebSocket conectado")

            # Mapeamento de eventos do websocket
            self.ws.on("tick", self._on_tick)
            self.ws.on("balance", self._on_balance)
            self.ws.on("trade_result", self._on_trade_result)

            # Inscrições Iniciais
            await self.ws.subscribe_ticks(settings.SYMBOL)
            await self.ws.subscribe_balance()
            
            # Inscreve-se explicitamente para acompanhar o PnL dos contratos abertos
            await self.ws._send({"proposal_open_contract": 1, "subscribe": 1})

            while self.running:
                await asyncio.sleep(0.5)
                
        except asyncio.CancelledError:
            self.emitter.log_message.emit("🔴 Loop async cancelado")
        except Exception as e:
            self.emitter.log_message.emit(f"❌ Erro crítico no Orquestrador: {e}")
            logger.error(f"Erro crítico: {e}")
        finally:
            await self.ws.close()
            self.emitter.connection_status.emit(False)

    async def _on_tick(self, data):
        if "tick" not in data:
            return
            
        quote = float(data["tick"]["quote"])
        self.prices = np.append(self.prices, quote)[-500:]
        
        if len(self.prices) >= 20:
            await self._execute_cycle()
        else:
            if len(self.prices) % 5 == 0:
                self.emitter.log_message.emit(f"📊 Coletando dados... {len(self.prices)}/20 ticks")

    async def _on_balance(self, data):
        if "balance" not in data:
            return
            
        bal = float(data["balance"]["balance"])
        self.state.update(self.risk.update(bal))
        self._update_ui_state()

    async def _on_trade_result(self, data):
        # 1. Confirmação de que a ordem de compra foi executada
        if "buy" in data:
            buy_info = data["buy"]
            self.trades_count += 1
            self.emitter.log_message.emit(f"📈 Contrato Aberto: ID {buy_info.get('contract_id', 'N/A')}")
            return

        # 2. Acompanhamento e finalização do contrato aberto
        if "proposal_open_contract" in data:
            contract = data["proposal_open_contract"]
            
            # Se is_sold for verdadeiro, o trade finalizou
            if contract.get("is_sold"):
                status = contract.get("status", "")
                profit = float(contract.get("profit", 0.0))
                
                if status == "won":
                    self.wins += 1
                    self.emitter.log_message.emit(f"✅ WIN | Lucro: +${profit:.2f}")
                elif status == "lost":
                    self.emitter.log_message.emit(f"❌ LOSS | Prejuízo: ${profit:.2f}")

                # Atualiza a gestão de risco e libera o sistema para um novo trade
                self.risk.log_trade(profit)
                self.in_trade = False
                self._update_ui_state()

    async def _execute_cycle(self):
        if self.risk.circuit_breaker:
            # Emite alerta apenas se estiver tentando processar algo, evita spam no log
            if not getattr(self, '_circuit_notified', False):
                self.emitter.log_message.emit("🛑 Circuit Breaker ATIVO - Pausa de Proteção")
                self._circuit_notified = True
            return

        self._circuit_notified = False
        sig, score, votes = self.consensus.compute(self.prices, self.state)
        self._update_ui_state(sig, score, votes)

        # Condição de entrada travada pela flag in_trade
        if sig in ("CALL", "PUT") and not self.in_trade:
            self.in_trade = True  # Trava imediata para não abrir operações duplicadas
            stake = self.risk.get_stake()
            
            self.emitter.log_message.emit(f"📤 Enviando {sig} | Stake: ${stake:.2f} | Confiança: {score:.2%}")
            
            try:
                await self.ws.execute_trade(sig, stake, settings.SYMBOL, settings.DURATION)
            except Exception as e:
                self.emitter.log_message.emit(f"⚠️ Erro ao enviar ordem: {e}")
                self.in_trade = False  # Libera a trava em caso de falha de requisição

    def _update_ui_state(self, signal="HOLD", score=0.0, votes=None):
        if votes is None:
            votes = {}
            
        pnl = self.risk.current_balance - self.risk.start_balance
        wr = (self.wins / self.trades_count * 100) if self.trades_count > 0 else 0.0
        
        self.emitter.tick_update.emit(
            self.prices.tolist(), signal, score, votes,
            self.state.get("drawdown", 0.0), pnl, self.trades_count, wr
        )