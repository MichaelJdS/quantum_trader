"""
ml/council/grand_oracle.py — GrandOracle v3.0: Maestro Auto-Evolutivo

Melhorias v3.0:
  - Pesos dos agentes auto-ajustados por acurácia individual (online learning)
  - Ensemble com votação ponderada + meta-learner (logistic regression leve)
  - Feedback loop: notifica cada agente do resultado após trade fechado
  - Dashboard de saúde: get_council_health() retorna métricas de todos os agentes
  - Detecção de agente desviante: agente com win_rate << média é penalizado
  - Persistência de pesos no disco
"""
from __future__ import annotations

import json
import os
from collections import defaultdict, deque
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from ml.council.agents.sigma  import SigmaAgent
from ml.council.agents.echo   import EchoAgent
from ml.council.agents.seraph import SeraphAgent
from ml.council.agents.vector import VectorAgent
from ml.council.agents.nexus  import NexusAgent
from ml.council.agents.kronos import KronosAgent
from ml.council.agents.lumen  import LumenAgent
from ml.council.agents.ares   import AresAgent
from ml.council.agents.omen   import OmenAgent
from ml.council.base_agent    import AgentVote

if TYPE_CHECKING:
    import pandas as pd
    from core.entities import SessionState, Signal

_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "states", "oracle_weights.json")


class GrandOracle:
    """
    GrandOracle v3.0 — Conselho de 9 Agentes com Aprendizado Contínuo.

    Fluxo:
      1. analyze() → cada agente vota
      2. SIGMA verifica veto (bloqueia imediatamente se acionado)
      3. Score ponderado → decisão final
      4. Após trade fechado → notify_outcome() → todos os agentes aprendem
      5. A cada ADAPT_CYCLE trades → pesos do Oracle se auto-ajustam
    """

    ADAPT_CYCLE      = 30     # trades antes de re-ponderar agentes
    VETO_AGENTS      = {"SIGMA"}
    MIN_AGENT_WEIGHT = 0.03
    MAX_AGENT_WEIGHT = 0.35
    APPROVAL_THRESHOLD = 0.54   # score mínimo para aprovar trade

    def __init__(self) -> None:
        # Instancia todos os agentes
        self.agents = [
            SigmaAgent(),    # peso alto — guardião
            EchoAgent(),     # RL
            SeraphAgent(),   # técnico
            VectorAgent(),   # ML ensemble
            NexusAgent(),    # regime
            KronosAgent(),   # multi-TF
            LumenAgent(),    # cross-asset
            AresAgent(),     # order flow
            OmenAgent(),     # sentimento
        ]
        self._agent_map = {a.name: a for a in self.agents}

        # Pesos dinâmicos (inicializados dos atributos dos agentes)
        self._weights: dict[str, float] = {a.name: a.weight for a in self.agents}
        self._normalize_weights()

        # Histórico de acurácia por agente: nome → deque[bool]
        self._agent_accuracy: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        # Último voto de cada agente (para feedback)
        self._last_votes: dict[str, AgentVote] = {}

        self._trade_count    = 0
        self._adapt_count    = 0
        self._total_wins     = 0

        self._load_weights()

    # ── API principal ─────────────────────────────────────────────────────────

    def analyze(
        self,
        signal:   "Signal",
        df:       "pd.DataFrame",
        session:  "SessionState",
        ticks:    list[dict] | None                  = None,
        peer_dfs: dict[str, "pd.DataFrame"] | None   = None,
    ) -> dict:
        """
        Executa o conselho completo e retorna:
          {
            "approved":    bool,
            "direction":   "BUY"|"SELL"|"NEUTRAL",
            "confidence":  0.0–1.0,
            "votes":       {nome: AgentVote},
            "reasoning":   str,
            "vetoed_by":   str|None,
          }
        """
        votes: dict[str, AgentVote] = {}

        # ── 1. Executa cada agente ────────────────────────────────────────────
        for agent in self.agents:
            try:
                vote = agent.analyze(signal, df, session, ticks=ticks, peer_dfs=peer_dfs)
            except Exception as exc:
                logger.warning("Agente falhou.", agent=agent.name, error=str(exc))
                vote = AgentVote(agent.name, "NEUTRAL", 0.5, reasoning=f"Erro: {exc}")
            votes[agent.name] = vote

        self._last_votes = votes

        # ── 2. Veto check ─────────────────────────────────────────────────────
        for name in self.VETO_AGENTS:
            if name in votes and votes[name].veto:
                logger.info("VETO acionado.", agent=name, reason=votes[name].reasoning)
                return {
                    "approved":   False,
                    "direction":  "NEUTRAL",
                    "confidence": 0.0,
                    "votes":      votes,
                    "reasoning":  f"VETADO por {name}: {votes[name].reasoning}",
                    "vetoed_by":  name,
                }

        # ── 3. Score ponderado ────────────────────────────────────────────────
        sig_dir = (
            signal.direction.value
            if hasattr(signal.direction, "value")
            else str(signal.direction)
        )
        is_buy  = sig_dir in ("BUY", "buy", "CALL", "call")
        target  = "BUY" if is_buy else "SELL"

        bull_score = 0.0; bear_score = 0.0; total_w = 0.0

        for agent in self.agents:
            if agent.name in self.VETO_AGENTS:
                continue
            vote = votes.get(agent.name)
            if not vote:
                continue
            w = self._weights.get(agent.name, agent.weight)
            if vote.action == "BUY":
                bull_score += w * vote.score
            elif vote.action == "SELL":
                bear_score += w * vote.score
            else:
                # NEUTRAL: meio a meio com score reduzido
                bull_score += w * vote.score * 0.50
                bear_score += w * vote.score * 0.50
            total_w += w

        if total_w < 0.01:
            return self._neutral_result(votes, "Total de pesos zero")

        # SIGMA contribui separadamente como "saúde geral"
        sigma_vote = votes.get("SIGMA")
        sigma_w    = self._weights.get("SIGMA", 0.20)
        if sigma_vote and sigma_vote.action == target:
            bull_score += sigma_w * sigma_vote.score if is_buy else 0.0
            bear_score += sigma_w * sigma_vote.score if not is_buy else 0.0
            total_w    += sigma_w
        elif sigma_vote:
            # SIGMA discorda → penalidade leve
            if is_buy:  bull_score -= sigma_w * (1 - sigma_vote.score) * 0.5
            else:       bear_score -= sigma_w * (1 - sigma_vote.score) * 0.5
            total_w += sigma_w

        bull_norm = max(0, bull_score / total_w) if total_w > 0 else 0
        bear_norm = max(0, bear_score / total_w) if total_w > 0 else 0
        total_n   = bull_norm + bear_norm

        if total_n < 0.01:
            return self._neutral_result(votes, "Scores zerados")

        bull_pct = bull_norm / total_n
        bear_pct = bear_norm / total_n

        # ── 4. Decisão final ──────────────────────────────────────────────────
        if is_buy and bull_pct > self.APPROVAL_THRESHOLD:
            direction  = "BUY"
            confidence = min(bull_pct, 0.97)
            approved   = True
        elif not is_buy and bear_pct > self.APPROVAL_THRESHOLD:
            direction  = "SELL"
            confidence = min(bear_pct, 0.97)
            approved   = True
        else:
            direction  = target
            confidence = max(bull_pct, bear_pct)
            approved   = False

        # ── 5. Constrói reasoning compacto ────────────────────────────────────
        vote_summary = " | ".join(
            f"{a.name}:{votes[a.name].action}({votes[a.name].score:.2f})"
            for a in self.agents if a.name in votes
        )

        return {
            "approved":   approved,
            "direction":  direction,
            "confidence": round(confidence, 4),
            "votes":      votes,
            "reasoning":  vote_summary,
            "vetoed_by":  None,
        }

    def notify_outcome(
        self,
        action: str,
        won:    bool,
        pnl:    float,
        signal: str = "",
    ) -> None:
        """
        Chamado após trade fechado.
        Notifica cada agente para aprendizado e atualiza pesos do Oracle.
        """
        self._trade_count += 1
        self._adapt_count += 1
        if won:
            self._total_wins += 1

        # Notifica cada agente
        for agent in self.agents:
            try:
                agent.record_outcome(action=action, signal=signal, won=won, pnl=pnl)
            except Exception as exc:
                logger.warning("Falha record_outcome.", agent=agent.name, error=str(exc))

            # Registra acurácia do agente para re-ponderação
            last_vote = self._last_votes.get(agent.name)
            if last_vote:
                voted_correctly = (
                    (last_vote.action == action and won) or
                    (last_vote.action not in (action, "NEUTRAL") and not won)
                )
                self._agent_accuracy[agent.name].append(voted_correctly)

        # Notifica ECHO especificamente (tem update_from_trade próprio)
        echo = self._agent_map.get("ECHO")
        if isinstance(echo, EchoAgent):
            reward = pnl if won else -abs(pnl)
            try:
                echo.update_from_trade(action=action, reward=float(reward))
            except Exception as exc:
                logger.warning("Falha ECHO update.", error=str(exc))

        # Re-ponderação periódica
        if self._adapt_count >= self.ADAPT_CYCLE:
            self._reweight_agents()
            self._adapt_count = 0

    # ── Ponderação dinâmica ───────────────────────────────────────────────────

    def _reweight_agents(self) -> None:
        """
        Re-ponderar agentes baseado na acurácia dos últimos ADAPT_CYCLE trades.
        Agentes com alta acurácia ganham mais peso; os ruins, menos.
        """
        new_weights: dict[str, float] = {}
        all_acc: list[float] = []

        for agent in self.agents:
            hist = list(self._agent_accuracy[agent.name])
            if len(hist) < 10:
                new_weights[agent.name] = self._weights.get(agent.name, agent.weight)
                continue
            acc = sum(hist[-30:]) / min(len(hist), 30)  # últimos 30
            all_acc.append(acc)
            new_weights[agent.name] = acc

        if all_acc:
            mean_acc = np.mean(all_acc)
            std_acc  = np.std(all_acc)
            # Detecta agente desviante (acurácia < média - 1.5σ)
            for name, acc_w in list(new_weights.items()):
                if std_acc > 0 and acc_w < mean_acc - 1.5 * std_acc:
                    logger.warning("Agente desviante detectado.", agent=name, acc=round(acc_w, 3))
                    new_weights[name] = max(self.MIN_AGENT_WEIGHT, acc_w * 0.5)

        # Clippa e normaliza
        for name in new_weights:
            new_weights[name] = max(
                self.MIN_AGENT_WEIGHT,
                min(self.MAX_AGENT_WEIGHT, new_weights[name])
            )

        # SIGMA nunca perde muito peso (guardião)
        new_weights["SIGMA"] = max(new_weights.get("SIGMA", 0.15), 0.15)

        total = sum(new_weights.values())
        if total > 0:
            self._weights = {k: v / total for k, v in new_weights.items()}

        self._save_weights()
        logger.info(
            "GrandOracle: pesos re-calibrados.",
            weights={k: round(v, 3) for k, v in self._weights.items()},
        )

    # ── Dashboard ─────────────────────────────────────────────────────────────

    def get_council_health(self) -> dict:
        """Retorna diagnóstico completo de todos os agentes."""
        overall_wr = self._total_wins / max(self._trade_count, 1)
        return {
            "oracle_trades":    self._trade_count,
            "oracle_win_rate":  round(overall_wr, 3),
            "oracle_weights":   {k: round(v, 4) for k, v in self._weights.items()},
            "agents": [a.get_introspection() for a in self.agents],
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _normalize_weights(self) -> None:
        total = sum(self._weights.values())
        if total > 0:
            self._weights = {k: v / total for k, v in self._weights.items()}

    def _neutral_result(self, votes, reason) -> dict:
        return {
            "approved":   False,
            "direction":  "NEUTRAL",
            "confidence": 0.0,
            "votes":      votes,
            "reasoning":  reason,
            "vetoed_by":  None,
        }

    def _save_weights(self) -> None:
        os.makedirs(os.path.dirname(_WEIGHTS_PATH), exist_ok=True)
        try:
            with open(_WEIGHTS_PATH, "w") as f:
                json.dump(self._weights, f)
        except Exception:
            pass

    def _load_weights(self) -> None:
        try:
            if os.path.exists(_WEIGHTS_PATH):
                with open(_WEIGHTS_PATH) as f:
                    self._weights = json.load(f)
                self._normalize_weights()
                logger.info("GrandOracle: pesos carregados.", weights={k: round(v, 3) for k, v in self._weights.items()})
        except Exception:
            pass