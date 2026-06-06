"""
ml/council/grand_oracle.py — Grand Oracle: Agregador do Conselho

Orquestra os 8 agentes especializados + Gemini Advisor.
Cada agente vota em paralelo. O Grand Oracle agrega via votação ponderada.
SIGMA tem poder de VETO absoluto.

Pesos (soma = 1.0):
  SIGMA  20%  SERAPH 13%  KRONOS 12%  VECTOR 12%
  NEXUS  10%  ARES   10%  GEMINI  8%  ECHO    8%  LUMEN   5%  (+ bônus alinhamento 2%)
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from ml.council.base_agent import AgentVote
from ml.council.agents.seraph  import SeraphAgent
from ml.council.agents.nexus   import NexusAgent
from ml.council.agents.kronos  import KronosAgent
from ml.council.agents.sigma   import SigmaAgent
from ml.council.agents.vector  import VectorAgent
from ml.council.agents.ares    import AresAgent
from ml.council.agents.echo    import EchoAgent
from ml.council.agents.lumen   import LumenAgent

if TYPE_CHECKING:
    from core.entities import Signal, SessionState
    from ml.gemini_advisor import GeminiAdvisor


# ── Resultado da deliberação do Conselho ──────────────────────────────────────

@dataclass
class CouncilDecision:
    """Resultado da votação do Oracle Council."""
    approved:            bool
    action:              str
    final_confidence:    float
    weighted_score:      float
    votes:               list[AgentVote]         = field(default_factory=list)
    veto_by:             str | None              = None
    reasoning:           str                     = ""
    gemini_reasoning:    str                     = ""
    timestamp:           float                   = field(default_factory=time.time)

    def summary(self) -> dict:
        return {
            "approved":         self.approved,
            "action":           self.action,
            "confidence":       round(self.final_confidence, 4),
            "weighted_score":   round(self.weighted_score, 4),
            "veto_by":          self.veto_by,
            "reasoning":        self.reasoning,
            "gemini_reasoning": self.gemini_reasoning,
            "votes": [
                {
                    "agent":     v.agent_name,
                    "action":    v.action,
                    "confidence": round(v.score, 4),
                    "veto":      v.veto,
                    "reasoning": v.reasoning,
                }
                for v in self.votes
            ],
        }


# ── Grand Oracle ──────────────────────────────────────────────────────────────

class GrandOracle:
    """
    Agrega os votos de todos os especialistas e toma a decisão final.
    Integra o GeminiAdvisor como o 9º conselheiro.

    Limiares de decisão:
      score ≥ 0.62 → APROVADO com confiança cheia
      score 0.50–0.62 → APROVADO com confiança reduzida (×0.85)
      score < 0.50 → BLOQUEADO
    """

    APPROVE_HIGH = 0.62
    APPROVE_LOW  = 0.50
    GEMINI_WEIGHT = 0.08

    def __init__(self, gemini_advisor: "GeminiAdvisor | None" = None) -> None:
        self._agents = [
            SigmaAgent(),
            SeraphAgent(),
            KronosAgent(),
            VectorAgent(),
            NexusAgent(),
            AresAgent(),
            EchoAgent(),
            LumenAgent(),
        ]
        self._gemini = gemini_advisor
        self._last_decision: CouncilDecision | None = None

    @property
    def echo_agent(self) -> EchoAgent:
        """Acesso direto ao ECHO para atualização pós-trade."""
        return next(a for a in self._agents if isinstance(a, EchoAgent))

    @property
    def last_decision(self) -> CouncilDecision | None:
        return self._last_decision

    # ── API Principal ─────────────────────────────────────────────────────────

    async def evaluate(
        self,
        signal: "Signal",
        df: pd.DataFrame,
        session: "SessionState",
        ticks: list[dict] | None = None,
        peer_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> CouncilDecision:
        """
        Avalia o sinal com todos os agentes e retorna CouncilDecision.
        Executa os agentes em paralelo via asyncio.
        """
        t0 = time.monotonic()

        # ── 1. Coleta votos de todos os agentes em paralelo ───────────────────
        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(
                None, agent.analyze, signal, df, session, ticks, peer_dfs
            )
            for agent in self._agents
        ]
        votes: list[AgentVote] = await asyncio.gather(*tasks)

        # ── 2. Voto do Gemini (usa cache — não faz chamada API aqui) ──────────
        gemini_vote, gemini_reasoning = self._get_gemini_vote(signal)
        if gemini_vote:
            votes.append(gemini_vote)

        sig_dir = signal.direction.value if hasattr(signal.direction, "value") else str(signal.direction)
        is_buy  = sig_dir in ("BUY", "buy", "CALL", "call")
        target_action = "BUY" if is_buy else "SELL"

        # ── 3. Verifica VETO ──────────────────────────────────────────────────
        for vote in votes:
            if vote.veto:
                decision = CouncilDecision(
                    approved=False,
                    action=target_action,
                    final_confidence=0.0,
                    weighted_score=0.0,
                    votes=list(votes),
                    veto_by=vote.agent_name,
                    reasoning=vote.reasoning,
                    gemini_reasoning=gemini_reasoning,
                )
                self._last_decision = decision
                self._log_decision(decision, signal, elapsed=time.monotonic() - t0)
                return decision

        # ── 4. Calcula score ponderado ────────────────────────────────────────
        weighted_score = self._compute_weighted_score(votes, target_action)

        # ── 5. Decisão final ──────────────────────────────────────────────────
        if weighted_score >= self.APPROVE_HIGH:
            approved = True
            final_confidence = signal.confidence * 1.10   # boost
            reasoning = f"Conselho aprova com alta confiança (score={weighted_score:.3f})"
        elif weighted_score >= self.APPROVE_LOW:
            approved = True
            final_confidence = signal.confidence * 0.85   # redução leve
            reasoning = f"Conselho aprova com confiança reduzida (score={weighted_score:.3f})"
        else:
            approved = False
            final_confidence = 0.0
            reasoning = f"Conselho bloqueia (score={weighted_score:.3f} < {self.APPROVE_LOW})"

        # Clamp confidence
        final_confidence = max(0.0, min(final_confidence, 1.0))

        decision = CouncilDecision(
            approved=approved,
            action=target_action if approved else "NEUTRAL",
            final_confidence=final_confidence,
            weighted_score=weighted_score,
            votes=list(votes),
            reasoning=reasoning,
            gemini_reasoning=gemini_reasoning,
        )
        self._last_decision = decision
        self._log_decision(decision, signal, elapsed=time.monotonic() - t0)
        return decision

    def update_echo_from_trade(self, action: str, pnl: float) -> None:
        """Notifica o ECHO sobre o resultado de um trade para aprendizado."""
        try:
            reward = 1.0 if pnl > 0 else -1.0
            self.echo_agent.update_from_trade(action, reward)
        except Exception:
            pass

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_gemini_vote(self, signal) -> tuple[AgentVote | None, str]:
        """Converte o último conselho do Gemini em um AgentVote."""
        if self._gemini is None or not self._gemini.is_enabled:
            return None, ""

        advice = self._gemini.last_advice
        if advice is None:
            return None, "Gemini: sem conselho ainda"

        sig_dir = signal.direction.value if hasattr(signal.direction, "value") else str(signal.direction)
        is_buy  = sig_dir in ("BUY", "buy", "CALL", "call")

        # Mapeia confidence_multiplier para score
        mult  = advice.confidence_multiplier
        score = max(0.3, min(0.9, 0.5 * mult))  # 1.0 → 0.50, 1.5 → 0.75

        if advice.risk_flag:
            # Gemini detectou risco → vota NEUTRAL com score baixo
            action = "NEUTRAL"
            score  = 0.35
        elif is_buy:
            action = "BUY"
        else:
            action = "SELL"

        vote = AgentVote(
            agent_name="GEMINI",
            action=action,
            score=score,
            reasoning=advice.reasoning[:80],
        )
        return vote, advice.reasoning

    def _compute_weighted_score(self, votes: list[AgentVote], target_action: str) -> float:
        """
        Calcula o score ponderado do Conselho com base em pesos fixos.
        Se um agente não votar, assume contribuição neutra (0.5).
        """
        weight_map = {
            "SIGMA":  0.20, "SERAPH": 0.13, "KRONOS": 0.12,
            "VECTOR": 0.12, "NEXUS":  0.10, "ARES":   0.10,
            "GEMINI": self.GEMINI_WEIGHT, "ECHO": 0.08, "LUMEN": 0.05,
        }

        total_weight = sum(weight_map.values())
        weighted_sum = 0.0

        vote_map = {v.agent_name: v for v in votes}

        for agent_name, w in weight_map.items():
            vote = vote_map.get(agent_name)

            if vote is None:
                contribution = 0.5
            elif vote.action == target_action:
                contribution = vote.score
            elif vote.action == "NEUTRAL":
                contribution = 0.5
            else:
                contribution = 1.0 - vote.score

            weighted_sum += w * contribution

        return weighted_sum / total_weight

    def _log_decision(self, decision: CouncilDecision, signal, elapsed: float) -> None:
        votes_str = " | ".join(
            f"{v.agent_name}:{v.action}({v.score:.2f})" for v in decision.votes
        )
        if decision.approved:
            logger.info(
                "🔮 Oracle Council APROVADO",
                symbol=signal.symbol,
                score=round(decision.weighted_score, 3),
                confidence=round(decision.final_confidence, 3),
                elapsed_ms=round(elapsed * 1000, 1),
                votes=votes_str,
            )
        else:
            logger.warning(
                "🔮 Oracle Council BLOQUEADO",
                symbol=signal.symbol,
                score=round(decision.weighted_score, 3),
                veto_by=decision.veto_by,
                reasoning=decision.reasoning,
                elapsed_ms=round(elapsed * 1000, 1),
            )

    def get_status(self) -> dict:
        """Retorna o status atual do Conselho para a API REST."""
        if self._last_decision is None:
            return {"status": "idle", "last_decision": None}
        return {
            "status": "active",
            "last_decision": self._last_decision.summary(),
        }
