import logging
from typing import Dict, Union

class KellyRiskManager:
    """
    Motor de Gerenciamento de Risco institucional utilizando 
    o Critério de Kelly Fracionado para sistemas de Alta Frequência.
    """
    
    def __init__(self, initial_bankroll: float, kelly_fraction: float = 0.25, max_daily_drawdown: float = 0.05):
        """
        Inicializa o gerenciador de risco.
        
        Args:
            initial_bankroll: Saldo inicial da conta.
            kelly_fraction: Multiplicador fracionário (padrão de 25% ou 0.25) para segurança.
            max_daily_drawdown: Limite rígido de perda diária (padrão de 5% ou 0.05).
        """
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.kelly_fraction = kelly_fraction
        self.max_daily_drawdown = max_daily_drawdown
        self.logger = logging.getLogger("KellyRiskManager")

    def calculate_stake(self, win_probability: float, payout_ratio: float = 0.95) -> Union[float, None]:
        """
        Calcula o tamanho exato da posição ($) baseado na probabilidade da IA.
        
        Args:
            win_probability: Probabilidade de acerto (p) gerada pela LSTM (ex: 0.58 para 58%).
            payout_ratio: Retorno da corretora em caso de acerto (b). Na Deriv, índices sintéticos 
                          geralmente pagam cerca de 95% do valor investido (0.95).
        
        Returns:
            Valor monetário da estaca (Stake) ou None se o risco não for validado.
        """
        # Proteção Primária: O modelo só deve operar se possuir um Edge (vantagem matemática)
        if win_probability <= 0.50:
            self.logger.warning(f"Edge negativo ({win_probability:.2%}). Operação abortada.")
            return None

        # Verificação do Limite de Drawdown (Circuit Breaker)
        if self._is_drawdown_limit_reached():
            return None

        p = win_probability
        q = 1.0 - p
        b = payout_ratio

        # Fórmula de Kelly: f* = (bp - q) / b
        kelly_percentage = (b * p - q) / b

        # Edge verificado pela fórmula
        if kelly_percentage <= 0:
            self.logger.warning("Fração de Kelly <= 0. Risco assimétrico desfavorável.")
            return None

        # Aplicação da Fração de Kelly para amortecimento de variância (Quarter-Kelly)
        safe_kelly_percentage = kelly_percentage * self.kelly_fraction
        
        # O tamanho da posição é a fração segura aplicada sobre o capital ATUAL
        stake = self.current_bankroll * safe_kelly_percentage
        
        # Limite máximo absoluto de segurança (Cap): Nunca arriscar mais que 2% em uma única entrada
        max_absolute_risk = self.current_bankroll * 0.02
        final_stake = min(stake, max_absolute_risk)

        # Arredondamento monetário (2 casas decimais)
        final_stake = round(final_stake, 2)
        
        self.logger.info(
            f"Probabilidade IA: {p:.2%} | Fração Alvo: {safe_kelly_percentage:.2%} | Estaca Recomendada: ${final_stake}"
        )
        return final_stake

    def update_bankroll(self, profit_loss: float) -> None:
        """
        Atualiza o saldo atualizado com base no resultado da última operação da Deriv.
        
        Args:
            profit_loss: Valor líquido ganho (+) ou perdido (-).
        """
        self.current_bankroll += profit_loss
        self.logger.info(f"Resultado do Trade: ${profit_loss:.2f} | Saldo Atual: ${self.current_bankroll:.2f}")

    def _is_drawdown_limit_reached(self) -> bool:
        """
        Calcula o Drawdown atual para interromper as operações caso o limite seja atingido.
        """
        current_drawdown = (self.initial_bankroll - self.current_bankroll) / self.initial_bankroll
        
        if current_drawdown >= self.max_daily_drawdown:
            self.logger.error(
                f"CIRCUIT BREAKER: Drawdown diário de {current_drawdown:.2%} excedeu o limite "
                f"({self.max_daily_drawdown:.2%}). Operações suspensas pelo sistema."
            )
            return True
            
        return False