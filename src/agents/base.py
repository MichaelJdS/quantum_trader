# src/agents/base.py
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Dict, Any
import numpy as np

class BaseAgent(ABC):
    name: str = "Base"
    weight: float = 1.0
    confidence: float = 0.5
    accuracy: float = 0.5
    trades: int = 0

    @abstractmethod
    def analyze(self, prices: np.ndarray, state: Dict[str, Any]) -> Tuple[Optional[str], float]:
        pass

    def update_metrics(self, pnl: float, regime: str):
        self.trades += 1
        win = 1 if pnl > 0 else 0
        self.accuracy = (self.accuracy * (self.trades - 1) + win) / self.trades
        self.confidence = 0.6 * self.accuracy + 0.4 * self.confidence * 0.95