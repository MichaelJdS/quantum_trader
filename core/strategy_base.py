from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from core.entities import RiskConfig, Signal, SessionState


class StrategyBase(ABC):
    """
    Contrato abstrato para todas as estratégias de trading.

    Qualquer nova estratégia deve herdar esta classe e implementar
    `generate_signal`. O framework cuida de risco, execução e logging.
    """

    def __init__(self, name: str, risk_config: RiskConfig) -> None:
        self.name = name
        self.risk_config = risk_config

    @abstractmethod
    def generate_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        session: SessionState,
    ) -> Signal | None:
        """
        Analisa o DataFrame de candles e retorna um Signal ou None.

        Args:
            df: DataFrame com colunas OHLCV + indicadores calculados.
            symbol: Símbolo sendo analisado (ex: "R_50").
            session: Estado atual da sessão (balance, win_rate, etc.).

        Returns:
            Signal se houver oportunidade, None caso contrário.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"