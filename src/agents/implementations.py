# src/agents/implementations.py
import numpy as np
from .base import BaseAgent
from .registry import register
from typing import Tuple, Optional, Dict, Any

def _ema(prices, period):
    alpha = 2 / (period + 1)
    return np.array([prices[0]] + [alpha * p + (1 - alpha) * prev for p, prev in zip(prices[1:], prices[:-1])])

@register
class TrendFollower(BaseAgent):
    name = "TrendFollower"
    def analyze(self, prices, state):
        if len(prices) < 20: return None, 0.5
        ema = _ema(prices, 14)[-1]
        sig = "CALL" if prices[-1] > ema else "PUT"
        conf = min(0.9, abs(prices[-1] - ema) / ema * 15 + 0.5)
        return sig, conf

@register
class MeanReversion(BaseAgent):
    name = "MeanReversion"
    def analyze(self, prices, state):
        if len(prices) < 30: return None, 0.5
        z = (prices[-1] - np.mean(prices[-20:])) / np.std(prices[-20:])
        if z > 1.5: return "PUT", 0.75
        if z < -1.5: return "CALL", 0.75
        return None, 0.5

@register
class VolatilityBreakout(BaseAgent):
    name = "VolatilityBreakout"
    def analyze(self, prices, state):
        if len(prices) < 25: return None, 0.5
        vol = np.std(prices[-10:])
        if vol > np.mean(prices[-20:]) * 0.002:
            return "CALL" if prices[-1] > prices[-2] else "PUT", 0.8
        return None, 0.5

@register
class MomentumScalper(BaseAgent):
    name = "MomentumScalper"
    def analyze(self, prices, state):
        if len(prices) < 15: return None, 0.5
        mom = np.sum(np.diff(prices[-8:]))
        return "CALL" if mom > 0 else "PUT", min(0.85, abs(mom) * 10 + 0.5)

@register
class PatternRecognizer(BaseAgent):
    name = "PatternRecognizer"
    def analyze(self, prices, state):
        if len(prices) < 5: return None, 0.5
        if prices[-3] < prices[-2] and prices[-2] > prices[-1]: return "PUT", 0.7
        if prices[-3] > prices[-2] and prices[-2] < prices[-1]: return "CALL", 0.7
        return None, 0.5

@register
class SentimentFlow(BaseAgent):
    name = "SentimentFlow"
    def analyze(self, prices, state):
        if len(prices) < 10: return None, 0.5
        ups = sum(1 for i in range(1, len(prices)) if prices[i] > prices[i-1])
        return "CALL" if ups > len(prices)/2 else "PUT", 0.6

@register
class Microstructure(BaseAgent):
    name = "Microstructure"
    def analyze(self, prices, state):
        if len(prices) < 20: return None, 0.5
        spreads = np.diff(prices[-10:])
        return "CALL" if np.sum(spreads[-5:]) > 0 else "PUT", min(0.85, abs(np.mean(spreads[-5:])) * 20 + 0.5)

@register
class RiskGuard(BaseAgent):
    name = "RiskGuard"
    def analyze(self, prices, state):
        if state.get("drawdown", 0) > settings.MAX_DRAWDOWN_PCT * 0.8: return "HOLD", 0.95
        if state.get("balance", 0) < 50: return "HOLD", 0.99
        return None, 0.5

@register
class ExecutionLatency(BaseAgent):
    name = "ExecutionLatency"
    def analyze(self, prices, state):
        return None, 0.6

@register
class AdaptiveLearner(BaseAgent):
    name = "AdaptiveLearner"
    def analyze(self, prices, state):
        return None, 0.65

@register
class PortfolioBalancer(BaseAgent):
    name = "PortfolioBalancer"
    def analyze(self, prices, state):
        return None, 0.5

@register
class QuantumTunnel(BaseAgent):
    name = "QuantumTunnel"
    def analyze(self, prices, state):
        if len(prices) < 15: return None, 0.5
        energy = abs(prices[-1] - np.mean(prices[-10:]))
        prob = 1 / (1 + np.exp(-energy * 8))
        return "CALL" if prob > 0.5 else "PUT", min(0.9, prob + 0.15)

@register
class RegimeDetector(BaseAgent):
    name = "RegimeDetector"
    def analyze(self, prices, state):
        if len(prices) < 30: return None, 0.5
        vol = np.std(prices[-20:])
        state["regime"] = "volatile" if vol > 1.5 else "trending" if vol > 0.8 else "ranging"
        return None, 0.8

@register
class NewsFilter(BaseAgent):
    name = "NewsFilter"
    def analyze(self, prices, state):
        return None, 0.5

@register
class ConsensusOrchestrator(BaseAgent):
    name = "ConsensusOrchestrator"
    def analyze(self, prices, state):
        return None, 0.5