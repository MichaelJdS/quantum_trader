"""
ml/council/agent_calibrator.py — Calibração Dinâmica de Pesos por Regime

Cada agente tem peso BASE fixo. O calibrador ajusta esses pesos
multiplicando por um fator de performance real por regime de mercado.
Usa Platt Scaling para calibrar a confiança bruta dos agentes.

Regime detectado pelo NEXUS (via grand_oracle). Pesos são recalculados
a cada 50 trades ou sempre que o regime muda.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from loguru import logger

Regime = Literal["TREND", "SIDEWAYS", "VOLATILE", "UNKNOWN"]

# ── Pesos base (deve somar 1.0) ───────────────────────────────────────────────
BASE_WEIGHTS: dict[str, float] = {
    "SIGMA":  0.20,
    "SERAPH": 0.13,
    "KRONOS": 0.12,
    "VECTOR": 0.12,
    "NEXUS":  0.10,
    "ARES":   0.10,
    "GEMINI": 0.08,
    "ECHO":   0.08,
    "LUMEN":  0.05,
}

# ── Multiplicadores por regime (ajuste empírico, editável) ────────────────────
REGIME_MULTIPLIERS: dict[Regime, dict[str, float]] = {
    "TREND": {
        "SIGMA":  1.1, "SERAPH": 1.2, "KRONOS": 1.0,
        "VECTOR": 1.3, "NEXUS":  0.9, "ARES":   1.1,
        "GEMINI": 1.0, "ECHO":   0.9, "LUMEN":  0.8,
    },
    "SIDEWAYS": {
        "SIGMA":  1.0, "SERAPH": 0.9, "KRONOS": 1.3,
        "VECTOR": 0.8, "NEXUS":  1.2, "ARES":   0.9,
        "GEMINI": 1.1, "ECHO":   1.1, "LUMEN":  1.0,
    },
    "VOLATILE": {
        "SIGMA":  1.3, "SERAPH": 0.8, "KRONOS": 1.0,
        "VECTOR": 0.9, "NEXUS":  1.0, "ARES":   1.2,
        "GEMINI": 0.9, "ECHO":   1.0, "LUMEN":  0.9,
    },
    "UNKNOWN": {name: 1.0 for name in BASE_WEIGHTS},
}


@dataclass
class AgentPerformance:
    """Histórico de performance de um agente num regime específico."""
    agent_name: str
    regime: Regime
    outcomes: deque = field(default_factory=lambda: deque(maxlen=100))
    # Parâmetros do Platt Scaling: confiança_calibrada = sigmoid(A * score + B)
    platt_A: float = 1.0
    platt_B: float = 0.0

    @property
    def win_rate(self) -> float:
        if not self.outcomes:
            return 0.5
        return sum(self.outcomes) / len(self.outcomes)

    @property
    def sample_size(self) -> int:
        return len(self.outcomes)

    def record(self, won: bool) -> None:
        self.outcomes.append(1.0 if won else 0.0)

    def fit_platt_scaling(self) -> None:
        """
        Ajusta parâmetros A e B do Platt Scaling via gradient descent simples.
        Requer pelo menos 20 amostras para ser significativo.
        """
        if self.sample_size < 20:
            return
        outcomes_arr = np.array(list(self.outcomes), dtype=float)
        # Usa win_rate como proxy de confiança bruta (simplificado)
        # Em produção, armazene os scores brutos junto com os outcomes.
        wr = self.win_rate
        if wr <= 0.0 or wr >= 1.0:
            return
        # Resolve log-odds: A*wr + B = log(wr/(1-wr))
        log_odds = math.log(wr / (1.0 - wr))
        self.platt_A = log_odds / max(wr, 1e-6)
        self.platt_B = 0.0
        logger.debug(
            "Platt Scaling atualizado.",
            agent=self.agent_name,
            regime=self.regime,
            A=round(self.platt_A, 4),
            B=round(self.platt_B, 4),
            wr=round(wr, 3),
        )

    def calibrate_score(self, raw_score: float) -> float:
        """Aplica Platt Scaling ao score bruto do agente."""
        logit = self.platt_A * raw_score + self.platt_B
        return 1.0 / (1.0 + math.exp(-logit))


class AgentCalibrator:
    """
    Gerencia pesos dinâmicos e calibração de confiança dos agentes do Conselho.

    Uso:
        calibrator = AgentCalibrator()
        weights = calibrator.get_weights(regime="TREND")
        cal_score = calibrator.calibrate("SERAPH", raw_score=0.72, regime="TREND")
        calibrator.record_outcome("SERAPH", won=True, regime="TREND")
    """

    RECALIBRATE_EVERY = 50  # trades por regime antes de re-fit do Platt

    def __init__(self) -> None:
        # performance[regime][agent_name]
        self._perf: dict[Regime, dict[str, AgentPerformance]] = defaultdict(dict)
        self._trade_count: dict[Regime, int] = defaultdict(int)
        self._current_regime: Regime = "UNKNOWN"
        self._computed_weights: dict[Regime, dict[str, float]] = {}

    # ── API pública ───────────────────────────────────────────────────────────

    def set_regime(self, regime: Regime) -> None:
        """Notifica mudança de regime. Recalcula pesos imediatamente."""
        if regime != self._current_regime:
            logger.info("Regime alterado.", old=self._current_regime, new=regime)
            self._current_regime = regime
            self._recompute_weights(regime)

    def get_weights(self, regime: Regime | None = None) -> dict[str, float]:
        """Retorna pesos normalizados para o regime atual (ou fornecido)."""
        r = regime or self._current_regime
        if r not in self._computed_weights:
            self._recompute_weights(r)
        return self._computed_weights[r]

    def calibrate(self, agent_name: str, raw_score: float, regime: Regime | None = None) -> float:
        """Retorna score calibrado via Platt Scaling para o agente/regime."""
        r = regime or self._current_regime
        perf = self._get_or_create_perf(agent_name, r)
        return perf.calibrate_score(raw_score)

    def record_outcome(self, agent_name: str, won: bool, regime: Regime | None = None) -> None:
        """Registra resultado de um trade para o agente no regime dado."""
        r = regime or self._current_regime
        perf = self._get_or_create_perf(agent_name, r)
        perf.record(won)
        self._trade_count[r] += 1

        # Re-fit Platt Scaling periodicamente
        if self._trade_count[r] % self.RECALIBRATE_EVERY == 0:
            for agent_perf in self._perf[r].values():
                agent_perf.fit_platt_scaling()
            self._recompute_weights(r)

    def get_agent_stats(self, regime: Regime | None = None) -> dict:
        """Retorna estatísticas de todos os agentes para o regime."""
        r = regime or self._current_regime
        weights = self.get_weights(r)
        result = {}
        for agent_name in BASE_WEIGHTS:
            perf = self._perf.get(r, {}).get(agent_name)
            result[agent_name] = {
                "weight": round(weights.get(agent_name, BASE_WEIGHTS[agent_name]), 4),
                "win_rate": round(perf.win_rate if perf else 0.5, 3),
                "samples": perf.sample_size if perf else 0,
                "platt_A": round(perf.platt_A if perf else 1.0, 4),
                "platt_B": round(perf.platt_B if perf else 0.0, 4),
            }
        return result

    # ── Internos ──────────────────────────────────────────────────────────────

    def _get_or_create_perf(self, agent_name: str, regime: Regime) -> AgentPerformance:
        if agent_name not in self._perf[regime]:
            self._perf[regime][agent_name] = AgentPerformance(
                agent_name=agent_name, regime=regime
            )
        return self._perf[regime][agent_name]

    def _recompute_weights(self, regime: Regime) -> None:
        """
        Recalcula pesos combinando:
          1. Peso base do agente
          2. Multiplicador do regime
          3. Fator de performance real (win_rate vs 0.5 baseline)
        """
        mults = REGIME_MULTIPLIERS.get(regime, REGIME_MULTIPLIERS["UNKNOWN"])
        raw: dict[str, float] = {}

        for agent_name, base_w in BASE_WEIGHTS.items():
            regime_mult = mults.get(agent_name, 1.0)
            perf = self._perf.get(regime, {}).get(agent_name)

            if perf and perf.sample_size >= 20:
                # Performance real: win_rate acima de 0.5 aumenta peso
                perf_factor = 0.5 + (perf.win_rate - 0.5) * 1.5
                perf_factor = max(0.3, min(2.0, perf_factor))
            else:
                perf_factor = 1.0

            raw[agent_name] = base_w * regime_mult * perf_factor

        # Normaliza para soma = 1.0
        total = sum(raw.values())
        self._computed_weights[regime] = {
            name: w / total for name, w in raw.items()
        }

        logger.debug(
            "Pesos recalculados.",
            regime=regime,
            weights={k: round(v, 4) for k, v in self._computed_weights[regime].items()},
        )


# Singleton global — importado pelo GrandOracle
_calibrator_instance: AgentCalibrator | None = None


def get_calibrator() -> AgentCalibrator:
    global _calibrator_instance
    if _calibrator_instance is None:
        _calibrator_instance = AgentCalibrator()
    return _calibrator_instance