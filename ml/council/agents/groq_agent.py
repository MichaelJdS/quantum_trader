"""
ml/council/agents/groq_agent.py — GroqAgent v1.0

Agente LLM para o Oracle Council.
Usa Groq (LLaMA-3.1-8b-instant) para analisar o contexto completo
e emitir voto BUY/SELL/NEUTRAL com score de convicção.
Timeout de 3s — nunca trava o conselho.
"""
from __future__ import annotations
import asyncio
import json
from typing import TYPE_CHECKING
from loguru import logger
from ml.council.base_agent import BaseAgent, AgentVote

if TYPE_CHECKING:
    import pandas as pd
    from core.entities import SessionState, Signal
    from ml.groq_engine import GroqEngine


SYSTEM_PROMPT = (
    "Você é GROQ, agente especialista em trading binário na Deriv. "
    "Analise os dados e responda APENAS com JSON válido, sem markdown: "
    '{"action": "BUY" ou "SELL" ou "NEUTRAL", "score": 0.0 a 1.0, "reason": "máx 10 palavras"}. '
    "Prefira NEUTRAL em caso de dúvida."
)


class GroqAgent(BaseAgent):
    name   = "GROQ"
    weight = 0.12

    def __init__(self, groq_engine: "GroqEngine") -> None:
        super().__init__()
        self._engine = groq_engine
        self._last_reasoning = ""

    def analyze(self, signal, df, session, ticks=None, peer_dfs=None) -> AgentVote:
        try:
            import concurrent.futures
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    self._async_analyze(signal, df, session), loop
                )
                return future.result(timeout=4.5)
            else:
                return loop.run_until_complete(self._async_analyze(signal, df, session))
        except Exception as exc:
            logger.warning("GroqAgent falhou.", error=str(exc))
            return self._neutral(f"Erro: {str(exc)[:60]}")

    async def _async_analyze(self, signal, df, session) -> AgentVote:
        prompt = self._build_prompt(signal, df, session)
        try:
            resp = await asyncio.wait_for(
                self._engine.complete(
                    system_prompt=SYSTEM_PROMPT,
                    user_message=prompt,
                    max_tokens=80,
                    temperature=0.1,
                    use_cache=False,
                ),
                timeout=3.0,
            )
        except asyncio.TimeoutError:
            logger.warning("GroqAgent: timeout 3s — votando NEUTRAL.")
            return self._neutral("Timeout LLM")

        if resp is None:
            return self._neutral("Groq sem resposta")
        return self._parse_response(resp.content)

    def _build_prompt(self, signal, df, session) -> str:
        direction = (
            signal.direction.value
            if hasattr(signal.direction, "value")
            else str(signal.direction)
        )
        rsi    = self._safe_float(df["rsi"].iloc[-1]       if "rsi"       in df.columns else None, 50.0)
        bb_pos = self._safe_float(df["bb_pos"].iloc[-1]    if "bb_pos"    in df.columns else None, 0.5)
        atr    = self._safe_float(df["atr"].iloc[-1]       if "atr"       in df.columns else None, 0.0)
        ema_s  = self._safe_float(df["ema_short"].iloc[-1] if "ema_short" in df.columns else None, 0.0)
        ema_l  = self._safe_float(df["ema_long"].iloc[-1]  if "ema_long"  in df.columns else None, 0.0)
        adx    = self._safe_float(df["adx"].iloc[-1]       if "adx"       in df.columns else None, 0.0)
        trend  = "ALTA" if ema_s > ema_l else "BAIXA" if ema_l > ema_s else "LATERAL"
        recent = list(self._memory)[-10:]
        rwr    = sum(1 for m in recent if m.won) / max(len(recent), 1) if recent else 0.5

        return (
            f"Sinal: {direction} | Estratégia: {getattr(signal, 'strategy_name', 'N/A')}\n"
            f"RSI: {rsi:.1f} | BB_pos: {bb_pos:.2f} | ATR: {atr:.5f} | ADX: {adx:.1f}\n"
            f"EMA tendência: {trend} (curta={ema_s:.4f} longa={ema_l:.4f})\n"
            f"Sessão — WR: {session.win_rate:.1%} | "
            f"Perdas consecutivas: {session.consecutive_losses} | "
            f"Trades: {session.total_trades} | Saldo: ${session.current_balance:.2f}\n"
            f"Meu histórico recente (10 trades): WR={rwr:.1%}\n"
            f"O sinal propõe {direction}. Você aprova?"
        )

    def _parse_response(self, content: str) -> AgentVote:
        try:
            raw = content.strip()
            if raw.startswith("```"):
                raw = "\n".join(l for l in raw.split("\n") if not l.startswith("```"))
            start, end = raw.find("{"), raw.rfind("}") + 1
            if start >= 0 and end > start:
                raw = raw[start:end]
            data   = json.loads(raw)
            action = str(data.get("action", "NEUTRAL")).upper()
            score  = max(0.0, min(1.0, float(data.get("score", 0.5))))
            reason = str(data.get("reason", ""))
            if action not in ("BUY", "SELL", "NEUTRAL"):
                action = "NEUTRAL"
            self._last_reasoning = reason
            logger.debug("GroqAgent votou.", action=action, score=round(score, 2))
            return AgentVote(agent_name=self.name, action=action, score=score, reasoning=reason)
        except Exception as exc:
            logger.warning("GroqAgent parse falhou.", error=str(exc))
            return self._neutral("Parse error")

    def _neutral(self, reason="N/A") -> AgentVote:
        return AgentVote(agent_name=self.name, action="NEUTRAL", score=0.5, reasoning=reason)

    def get_introspection(self) -> dict:
        base = super().get_introspection()
        base["groq_stats"]      = self._engine.get_stats() if self._engine else {}
        base["last_reasoning"]  = self._last_reasoning[:100]
        return base