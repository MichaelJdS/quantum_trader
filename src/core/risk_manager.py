import logging
import math
from typing import Union, Optional

class OnlineGARCH11:
    """
    Motor preditivo de volatilidade HFT utilizando GARCH(1,1).
    Processamento online O(1) por tick, evitando recálculos de arrays inteiros
    para manter a latência próxima a zero no WebSocket.
    """
    def __init__(self, omega: float = 0.000001, alpha: float = 0.05, beta: float = 0.94):
        # Os parâmetros clássicos do RiskMetrics aproximam alpha + beta = ~0.99
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        
        # Variância incondicional teórica inicial
        self.current_variance = omega / (1.0 - alpha - beta) if (alpha + beta) < 1.0 else 0.0001
        self.last_price: Optional[float] = None
        self.baseline_volatility: Optional[float] = None
        self.tick_count = 0

    def update_and_predict(self, current_price: float) -> float:
        """
        Atualiza o modelo com o novo preço e prevê a volatilidade instantânea.
        """
        if self.last_price is None:
            self.last_price = current_price
            return math.sqrt(self.current_variance)

        # Retorno logarítmico contínuo (epsilon)
        log_return = math.log(current_price / self.last_price)
        self.last_price = current_price

        # Atualização da equação GARCH(1,1)
        self.current_variance = self.omega + (self.alpha * (log_return ** 2)) + (self.beta * self.current_variance)
        current_volatility = math.sqrt(self.current_variance)

        # Calibração do baseline (média histórica de volatilidade normal) nos primeiros 100 ticks
        self.tick_count += 1
        if self.tick_count < 100:
            if self.baseline_volatility is None:
                self.baseline_volatility = current_volatility
            else:
                self.baseline_volatility = (self.baseline_volatility * 0.99) + (current_volatility * 0.01)

        return current_volatility


class KellyRiskManager:
    """
    Motor de Gerenciamento de Risco Institucional:
    Kelly Criterion fracionado + Amortecimento Dinâmico por GARCH(1,1).
    """
    def __init__(self, initial_bankroll: float, kelly_fraction: float = 0.25, max_daily_drawdown: float = 0.05):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.base_kelly_fraction = kelly_fraction
        self.max_daily_drawdown = max_daily_drawdown
        self.logger = logging.getLogger("GARCH-RiskManager")
        
        # Instância do Preditor de Volatilidade
        self.volatility_engine = OnlineGARCH11()

    def calculate_stake(self, win_probability: float, current_price: float, payout_ratio: float = 0.95) -> Union[float, None]:
        """
        Calcula o tamanho da posição ajustado dinamicamente pela volatilidade instantânea.
        """
        # 1. Atualização do GARCH(1,1) a cada avaliação
        current_vol = self.volatility_engine.update_and_predict(current_price)
        baseline_vol = self.volatility_engine.baseline_volatility or current_vol

        # 2. Proteção Primária e Drawdown Limit
        if win_probability <= 0.50:
            return None
        if self._is_drawdown_limit_reached():
            return None

        # 3. Fórmula de Kelly Clássica: f* = (bp - q) / b
        p = win_probability
        q = 1.0 - p
        b = payout_ratio
        kelly_percentage = (b * p - q) / b

        if kelly_percentage <= 0:
            return None

        # 4. Amortecimento Dinâmico GARCH (O Escudo Matemático)
        # Se a volatilidade atual for o dobro da baseline, o multiplicador cai pela metade
        volatility_penalty = baseline_vol / current_vol if current_vol > baseline_vol else 1.0
        
        # Limita a penalidade para evitar apostas irrisórias (cap de segurança)
        volatility_penalty = max(0.2, min(volatility_penalty, 1.0))

        # Kelly Fracionado dinamicamente ajustado
        dynamic_kelly_fraction = self.base_kelly_fraction * volatility_penalty
        safe_kelly_percentage = kelly_percentage * dynamic_kelly_fraction
        
        # 5. Cálculo do Stake final
        stake = self.current_bankroll * safe_kelly_percentage
        
        # Cap absoluto de exposição: Nunca arriscar mais que 2% (Circuit Breaker local)
        max_absolute_risk = self.current_bankroll * 0.02
        final_stake = round(min(stake, max_absolute_risk), 2)
        
        # Logs de auditoria para Grafana/Console
        if volatility_penalty < 0.8:
            self.logger.warning(
                f"🚨 ALERTA GARCH: Volatilidade anormal detectada! (Pico: {current_vol:.5f} vs Base: {baseline_vol:.5f}). "
                f"Multiplicador Kelly reduzido em {(1.0 - volatility_penalty):.1%}. Estaca protegida: ${final_stake}"
            )

        return final_stake

    def update_bankroll(self, profit_loss: float) -> None:
        self.current_bankroll += profit_loss
        self.logger.info(f"Resultado PnL: ${profit_loss:.2f} | Saldo Atual: ${self.current_bankroll:.2f}")

    def _is_drawdown_limit_reached(self) -> bool:
        current_drawdown = (self.initial_bankroll - self.current_bankroll) / self.initial_bankroll
        if current_drawdown >= self.max_daily_drawdown:
            self.logger.error(
                f"🛑 CIRCUIT BREAKER: Drawdown diário de {current_drawdown:.2%} excedeu o limite "
                f"({self.max_daily_drawdown:.2%}). Operações de entrada bloqueadas."
            )
            return True
        return False