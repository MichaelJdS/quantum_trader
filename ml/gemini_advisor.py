"""
ml/gemini_advisor.py — Consultor de Estratégias via Gemini

Usa o google-generativeai (Gemini 1.5 Flash, free tier) como um consultor
que analisa o contexto do mercado e recomenda qual estratégia priorizar,
ajuste de confiança e alertas de risco.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from core.settings import get_settings

if TYPE_CHECKING:
    from core.entities import SessionState

# Importação condicional para não travar se a lib não estiver instalada
try:
    from google import genai
    from google.genai import types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False
    logger.warning("google-genai não instalado. GeminiAdvisor desativado.")


# ── Tipos de dados ────────────────────────────────────────────────────────────

@dataclass
class GeminiAdvice:
    """Resposta estruturada do Gemini ao Orquestrador."""
    recommended_strategy: str           # ex: "ema_rsi_macd"
    confidence_multiplier: float        # ex: 1.15 → aumenta confiança em 15%
    risk_flag: bool                     # True → Gemini detectou risco elevado
    reasoning: str                      # Justificativa em texto para o log
    raw_response: str = ""              # Resposta bruta da API


@dataclass
class StrategyContext:
    """Contexto enviado ao Gemini para análise."""
    symbol: str
    candles_summary: str                # Últimos N candles como CSV resumido
    indicators: dict                    # Últimos valores dos indicadores
    session_state_summary: dict         # Balance, win_rate, consecutive_losses etc.
    available_strategies: list[str]     # Estratégias registradas no Engine
    last_strategy_results: dict         # Nome → taxa de acerto recente


# ── Classe Principal ──────────────────────────────────────────────────────────

class GeminiAdvisor:
    """
    Consultor de Estratégias baseado no Gemini 1.5 Flash (free tier).

    Integrado ao ExecutionEngine, é consultado a cada N ticks para decidir
    qual estratégia deve ser priorizada e se há alertas de risco.

    Configuração (via .env ou env vars):
        GEMINI_API_KEY  — chave da API do Google AI Studio (gratuita)
        GEMINI_MODEL    — modelo a usar (default: gemini-1.5-flash)
        GEMINI_CONSULT_INTERVAL_SECONDS — intervalo mínimo entre consultas em segundos (default: 300)
    """

    SYSTEM_INSTRUCTION = """
Você é um analista quantitativo. Responda SEMPRE e SOMENTE com JSON válido.
Nunca adicione texto, markdown, blocos de código ou explicações fora do JSON.
Formato obrigatório:
{
  "recommended_strategy": "BUY" | "SELL" | "NEUTRAL",
  "confidence_multiplier": <float entre 0.5 e 1.5>,
  "risk_flag": <true | false>,
  "reasoning": "<máximo 100 caracteres>"
}
"""

    def __init__(self) -> None:
        _s = get_settings()
        self._api_key    = _s.gemini_api_key
        self._model_name = _s.gemini_model
        self._interval   = _s.gemini_consult_interval_seconds
        self._last_consulted: float = 0.0
        self._last_advice: GeminiAdvice | None = None
        self._client = None
        self._enabled = False
        self._lock = asyncio.Lock()

        if not _GENAI_AVAILABLE:
            logger.warning("GeminiAdvisor: biblioteca não disponível.")
            return

        if not self._api_key:
            logger.warning(
                "GeminiAdvisor: GEMINI_API_KEY não configurada. "
                "Defina no .env para ativar o consultor."
            )
            return

        try:
            self._client = genai.Client(api_key=self._api_key)
            self._enabled = True
            logger.success(
                "GeminiAdvisor inicializado.",
                model=self._model_name,
                interval_s=self._interval,
            )
        except Exception as exc:
            logger.error("GeminiAdvisor: falha na inicialização.", error=str(exc))

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def last_advice(self) -> GeminiAdvice | None:
        return self._last_advice

    def should_consult(self) -> bool:
        """Retorna True se já passou o intervalo mínimo desde a última consulta."""
        if not self._enabled:
            return False
        elapsed = time.monotonic() - self._last_consulted
        return elapsed >= self._interval

    # ── API Pública ───────────────────────────────────────────────────────────

    async def consult(self, context: StrategyContext) -> GeminiAdvice | None:
        """
        Consulta o Gemini com o contexto do mercado.
        Retorna GeminiAdvice ou None se não for hora de consultar / falha.
        """
        if not self.should_consult():
            return self._last_advice

        async with self._lock:
            if not self.should_consult():  # double-check sob o lock
                return self._last_advice

            try:
                advice = await asyncio.to_thread(self._sync_consult, context)
                self._last_advice = advice
                self._last_consulted = time.monotonic()
                logger.info(
                    "Gemini Advisor consultado.",
                    symbol=context.symbol,
                    strategy=advice.recommended_strategy,
                    multiplier=advice.confidence_multiplier,
                    risk=advice.risk_flag,
                    reason=advice.reasoning[:80],
                )
                return advice
            except Exception as exc:
                logger.error("GeminiAdvisor: falha na consulta.", error=str(exc))
                return self._last_advice

    async def chat(self, user_message: str, context: str = "") -> str:
        """
        Interface de chat livre para o usuário conversar com o Gemini
        diretamente pelo painel do App Windows.
        """
        if not self._enabled or self._client is None:
            return "❌ Gemini não configurado. Adicione GEMINI_API_KEY ao .env"

        prompt = f"{context}\n\nPergunta do trader: {user_message}" if context else user_message
        try:
            response = await asyncio.to_thread(
                lambda: self._client.models.generate_content(  # type: ignore[union-attr]
                    model=self._model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=1024,
                    ),
                )
            )
            return response.text
        except Exception as exc:
            logger.error("GeminiAdvisor chat: falha.", error=str(exc))
            return f"❌ Erro ao consultar Gemini: {exc}"

    # ── Internals ─────────────────────────────────────────────────────────────

    def _sync_consult(self, context: StrategyContext) -> GeminiAdvice:
        """Chamada síncrona à API Gemini (executada via thread pool)."""
        prompt = self._build_prompt(context)
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                system_instruction=self.SYSTEM_INSTRUCTION,
                temperature=0.1,
                max_output_tokens=256,
            ),
        )
        raw = response.text.strip()
        return self._parse_response(raw, context.available_strategies)

    def _build_prompt(self, ctx: StrategyContext) -> str:
        return (
            f"Símbolo: {ctx.symbol}\n\n"
            f"Últimos Candles (OHLCV resumido):\n{ctx.candles_summary}\n\n"
            f"Indicadores Técnicos:\n{json.dumps(ctx.indicators, indent=2)}\n\n"
            f"Estado da Sessão:\n{json.dumps(ctx.session_state_summary, indent=2)}\n\n"
            f"Estratégias disponíveis: {ctx.available_strategies}\n\n"
            f"Resultados recentes por estratégia:\n{json.dumps(ctx.last_strategy_results, indent=2)}\n\n"
            "Analise e retorne o JSON de recomendação."
        )

    def _parse_response(self, raw: str, available: list[str]) -> GeminiAdvice:
        """Faz parsing do JSON retornado pelo Gemini com fallback seguro."""
        import re

        clean = raw.replace("```json", "").replace("```", "").strip()

        data = None
        try:
            data = json.loads(clean)
        except (json.JSONDecodeError, ValueError):
            # Fallback 1: regex
            match = re.search(r'\{[^{}]*"recommended_strategy"[^{}]*\}', clean, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except (json.JSONDecodeError, ValueError):
                    pass
            
            # Fallback 2: Tenta extrair primeiro/último {}
            if not data:
                try:
                    start = clean.index("{")
                    end = clean.rindex("}") + 1
                    data = json.loads(clean[start:end])
                except (ValueError, json.JSONDecodeError):
                    pass

        if data:
            strategy = data.get("recommended_strategy", "")
            if strategy not in available and available:
                strategy = available[0]
            return GeminiAdvice(
                recommended_strategy=strategy,
                confidence_multiplier=float(data.get("confidence_multiplier", 1.0)),
                risk_flag=bool(data.get("risk_flag", False)),
                reasoning=str(data.get("reasoning", "")),
                raw_response=raw,
            )

        # Fallback 3: Texto livre
        text_lower = raw.lower()
        if 'sell' in text_lower or 'bear' in text_lower or 'down' in text_lower:
            action = 'SELL'
        elif 'buy' in text_lower or 'bull' in text_lower or 'up' in text_lower:
            action = 'BUY'
        else:
            action = 'NEUTRAL'
            
        strategy = available[0] if available else action
            
        logger.warning(
            "GeminiAdvisor: falha no parsing JSON da resposta, usando heurística de texto livre.",
            action=action,
            raw_preview=raw[:200],
        )
        return GeminiAdvice(
            recommended_strategy=strategy,
            confidence_multiplier=1.0,
            risk_flag=False,
            reasoning=raw[:200],
            raw_response=raw,
        )


# ── Factory Helper ────────────────────────────────────────────────────────────

def build_context_from_df(
    df: pd.DataFrame,
    symbol: str,
    session_state: "SessionState",
    available_strategies: list[str],
    strategy_results: dict | None = None,
) -> StrategyContext:
    """
    Utilitário para construir um StrategyContext a partir do DataFrame de features.
    Chamado pelo ExecutionEngine antes de consultar o Gemini.
    """
    n = min(30, len(df))
    recent = df.tail(n)

    # Resumo CSV compacto dos últimos candles
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in recent.columns]
    candles_summary = recent[cols].round(5).to_csv(index=False)

    # Últimos valores dos indicadores técnicos relevantes
    last_row = df.iloc[-1]
    indicators: dict = {}
    for col in ["ema_fast", "ema_slow", "rsi", "macd", "bb_upper", "bb_lower", "atr", "adx"]:
        if col in last_row.index:
            val = last_row[col]
            indicators[col] = round(float(val), 6) if pd.notna(val) else None

    session_summary = {
        "balance": round(session_state.current_balance, 2),
        "initial_balance": round(session_state.initial_balance, 2),
        "win_rate": round(session_state.win_rate, 4),
        "total_trades": session_state.total_trades,
        "consecutive_losses": session_state.consecutive_losses,
    }

    return StrategyContext(
        symbol=symbol,
        candles_summary=candles_summary,
        indicators=indicators,
        session_state_summary=session_summary,
        available_strategies=available_strategies,
        last_strategy_results=strategy_results or {},
    )
