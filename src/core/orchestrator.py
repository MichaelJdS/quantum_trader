import os
import json
import asyncio
import logging
import websockets
import numpy as np
import pandas as pd
import torch
import joblib
from collections import deque
from typing import Optional

# Importando os motores matemáticos da nossa infraestrutura
from src.models.lstm_predictor import MarketLSTM
from src.core.risk_manager import KellyRiskManager

# Configuração de Logs Institucionais
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s - %(message)s"
)
logger = logging.getLogger("DerivOrchestrator")

class DerivOrchestrator:
    """
    Motor de Execução HFT Definitivo para a corretora Deriv.
    Gestão de Risco GARCH(1,1) + IA Preditiva + Execução de Contratos Real-time.
    """
    def __init__(self, app_id: str, api_token: str, symbol: str = "R_100"):
        self.app_id = app_id
        self.api_token = api_token
        self.symbol = symbol
        self.ws_url = f"wss://ws.binaryws.com/websockets/v3?app_id={self.app_id}"
        
        self.ws = None # Objeto de conexão WebSocket global
        self.device = torch.device("cpu")
        self.scaler = None
        
        # Banca inicial base de configuração do Kelly (Será sincronizada via API posteriormente)
        self.risk_manager = KellyRiskManager(initial_bankroll=1000.0, kelly_fraction=0.25, max_daily_drawdown=0.05)
        self.tick_buffer = deque(maxlen=250)
        
        # O Modelo é carregado após a inicialização das variáveis acima
        self.model = self._load_model()
        
        # Controle de Estado e Travas (Locks)
        self.is_authorized = False
        self.in_trade = False
        
    def _load_model(self) -> MarketLSTM:
        """Carrega a Super IA e o Normalizador Estatístico (Scaler)."""
        model_path = "models/best_lstm.pth"
        scaler_path = "models/scaler.pkl"
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("⚠️ Modelo ou Scaler não encontrados na pasta 'models/'. Treine o modelo primeiro.")
            
        # Carrega o Normalizador Z-Score
        self.scaler = joblib.load(scaler_path)
        
        # ⚠️ ARQUITETURA CALIBRADA COM O RESULTADO DO TREINO (41 Neurônios, 2 Camadas)
        model = MarketLSTM(input_size=8, hidden_size=41, num_layers=2, dropout=0.0)
        
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval() # Desliga o dropout e fixa os pesos para inferência HFT
        
        logger.info("🧠 Cérebro Quântico (LSTM) e Scaler carregados para inferência de Produção.")
        return model

    def _extract_live_features(self) -> Optional[torch.Tensor]:
        """Engenharia de features em tempo real com normalização Zero-Latency."""
        if len(self.tick_buffer) < 250:
            return None
            
        df = pd.DataFrame(list(self.tick_buffer), columns=['price'])
        close = df['price']
        
        # Indicadores Matemáticos
        df['log_return'] = np.log(close / close.shift(1))
        
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        sma_20 = close.rolling(window=20).mean()
        std_20 = close.rolling(window=20).std()
        df['bb_upper_dist'] = (sma_20 + (std_20 * 2)) - close
        df['bb_lower_dist'] = close - (sma_20 - (std_20 * 2))
        df['volatility_30'] = df['log_return'].rolling(window=30).std()
        
        df.dropna(inplace=True)
        
        # Filtra estritamente as colunas usadas no treino e os últimos 180 ticks
        feature_cols = ['log_return', 'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'bb_upper_dist', 'bb_lower_dist', 'volatility_30']
        recent_data = df[feature_cols].values[-180:]
        
        # Aplica a Normalização Treinada
        recent_data_scaled = self.scaler.transform(recent_data)
        
        return torch.tensor(recent_data_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

    async def _handle_prediction_and_trade(self, current_price: float):
        """Analisa o gráfico, infere predições e passa pela guilhotina de risco GARCH."""
        tensor_x = self._extract_live_features()
        if tensor_x is None:
            return
            
        with torch.inference_mode():
            output = self.model(tensor_x)
            probabilities = torch.softmax(output, dim=1)[0]
            prob_put, prob_call = probabilities[0].item(), probabilities[1].item()

        is_call = prob_call > prob_put
        max_prob = prob_call if is_call else prob_put
        direction_str = "CALL 🟢" if is_call else "PUT 🔴"

        # O Gestor de Risco (GARCH + Kelly) decide o tamanho da estaca com base na confiança
        stake = self.risk_manager.calculate_stake(max_prob, current_price, 0.95)

        if stake is not None and not self.in_trade:
            logger.warning(f"⚡ SINAL DETECTADO: {direction_str} | Confiança: {max_prob:.2%} | Stake Autorizado: ${stake:.2f}")
            await self._execute_trade(stake, is_call)

    async def _execute_trade(self, amount: float, is_call: bool):
        """Passo 1: Solicita uma proposta de contrato oficial da Deriv."""
        self.in_trade = True
        contract_type = "CALL" if is_call else "PUT"
        
        logger.info(f"🚀 Enviando solicitação de Proposta para {contract_type} de ${amount}...")
        
        proposal_req = {
            "proposal": 1,
            "amount": amount,
            "basis": "stake",
            "contract_type": contract_type,
            "currency": "USD",
            "duration": 1,
            "duration_unit": "m",
            "symbol": self.symbol,
            "req_id": 100 # Identificador único do bot para propostas
        }
        await self.ws.send(json.dumps(proposal_req))

    async def _monitor_contract(self, contract_id: int):
        """Passo 3: Aguarda a vela de 1 minuto fechar e extrai o PnL (Lucro/Prejuízo)."""
        logger.info("⏳ Monitorando operação. Aguardando 60 segundos de fechamento da vela...")
        await asyncio.sleep(62) # 60 segundos de contrato + 2 seg de delay de liquidação da corretora
        
        req = {
            "proposal_open_contract": 1,
            "contract_id": contract_id,
            "req_id": 102 # Identificador único do bot para auditoria
        }
        await self.ws.send(json.dumps(req))

    async def start(self):
        """Loop de Sustentação Resiliente HFT."""
        logger.info(f"🌐 Iniciando Motor HFT na Deriv (Ativo: {self.symbol})...")
        
        # Loop Externo de Reconexão (Se a internet da VPS piscar, o bot reinicia sozinho)
        while True:
            try:
                async with websockets.connect(self.ws_url, ping_interval=30) as websocket:
                    self.ws = websocket
                    self.is_authorized = False
                    self.in_trade = False
                    self.tick_buffer.clear()
                    
                    # 1. Autenticação na API
                    await self.ws.send(json.dumps({"authorize": self.api_token}))
                    
                    # 2. Assinatura de Ticks do Ativo
                    await self.ws.send(json.dumps({"ticks": self.symbol, "subscribe": 1}))
                    
                    # 3. Ouvinte Global Assíncrono (Event Loop)
                    while True:
                        response = await self.ws.recv()
                        data = json.loads(response)
                        msg_type = data.get("msg_type")

                        if "error" in data:
                            logger.error(f"❌ Erro da API ({msg_type}): {data['error']['message']}")
                            if data.get("req_id") in [100, 101, 102]:
                                self.in_trade = False
                            continue

                        # --- Roteamento de Mensagens ---
                        if msg_type == "authorize":
                            self.is_authorized = True
                            # Sincroniza a banca do bot com o saldo real lido na conta!
                            self.risk_manager.current_bankroll = float(data["authorize"]["balance"])
                            self.risk_manager.initial_bankroll = self.risk_manager.current_bankroll
                            logger.info(f"✅ Conexão Autorizada. Saldo Sincronizado: ${self.risk_manager.current_bankroll:.2f}")

                        elif msg_type == "tick":
                            current_price = float(data["tick"]["quote"])
                            self.tick_buffer.append(current_price)
                            
                            if len(self.tick_buffer) % 50 == 0:
                                logger.info(f"📈 Price: {current_price} | Buffer: {len(self.tick_buffer)}/250")
                                
                            if self.is_authorized:
                                asyncio.create_task(self._handle_prediction_and_trade(current_price))

                        # Resposta à solicitação _execute_trade
                        elif msg_type == "proposal" and data.get("req_id") == 100:
                            proposal_id = data["proposal"]["id"]
                            ask_price = data["proposal"]["ask_price"]
                            
                            logger.info(f"✅ Proposta validada. Disparando compra a ${ask_price} (Zero-Latency)...")
                            # Passo 2: Executa a compra imediatamente
                            buy_req = {"buy": proposal_id, "price": ask_price, "req_id": 101}
                            await self.ws.send(json.dumps(buy_req))

                        # Confirmação de que o contrato entrou
                        elif msg_type == "buy" and data.get("req_id") == 101:
                            contract_id = data["buy"]["contract_id"]
                            logger.warning(f"🛒 ORDEM EXECUTADA! ID: {contract_id}. O mercado está rolando...")
                            asyncio.create_task(self._monitor_contract(contract_id))

                        # Resultado final extraído após os 60 segundos
                        elif msg_type == "proposal_open_contract" and data.get("req_id") == 102:
                            contract = data["proposal_open_contract"]
                            status = contract["status"]
                            profit = float(contract["profit"])
                            
                            if status == "won":
                                logger.info(f"🏆 WIN! Operação encerrada com Lucro: +${profit:.2f}")
                            else:
                                logger.info(f"💀 LOSS. Operação encerrada com Prejuízo: -${abs(profit):.2f}")
                                
                            # Atualiza o saldo e calibra o Risco e a Volatilidade GARCH para a próxima operação
                            self.risk_manager.update_bankroll(profit)
                            self.in_trade = False

            except websockets.exceptions.ConnectionClosed:
                logger.warning("⚠️ Conexão WebSocket interrompida. O HFT fará reconexão automática em 3 segundos...")
                await asyncio.sleep(3)
            except Exception as e:
                logger.critical(f"❌ Erro de Sistema: {e}. Reiniciando túnel WebSocket...", exc_info=True)
                await asyncio.sleep(5)