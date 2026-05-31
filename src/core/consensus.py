import numpy as np
from typing import Tuple, Dict, Any
from src.agents import instantiate_all  # ✅ Importe assim, não de .registry
from src.models.quantum_optimizer import quantum_weight_optimizer
from src.config import settings

class ConsensusEngine:
    def __init__(self):
        self.agents = instantiate_all()
        self.tick_counter = 0

    def compute(self, prices: np.ndarray, state: Dict[str, Any]) -> Tuple[str, float, Dict[str, Dict]]:
        self.tick_counter += 1
        regime = state.get("regime", "neutral")
        
        # Recalcula pesos quânticos a cada 50 ticks
        if self.tick_counter % 50 == 0:
            weights = quantum_weight_optimizer([a.confidence for a in self.agents], regime)
            for a, w in zip(self.agents, weights): a.weight = w

        votes = {}
        signals = {"CALL": 0.0, "PUT": 0.0, "HOLD": 0.0}
        total_w = 0.0

        for agent in self.agents:
            sig, conf = agent.analyze(prices, state)
            if not sig: continue
            w = agent.weight * conf
            signals[sig] += w
            total_w += w
            votes[agent.name] = {"signal": sig, "conf": conf, "w": w}

        if total_w == 0: return "HOLD", 0.0, votes
        final = max(signals, key=signals.get)
        score = signals[final] / total_w
        return (final if score >= settings.CONSENSUS_THRESHOLD else "HOLD"), score, votes