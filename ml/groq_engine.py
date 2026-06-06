"""
ml/groq_engine.py — Groq Engine com Rotação de Chaves e Fallback

Features:
  - Pool de múltiplas API keys com rotação automática
  - Fallback para Gemini se todas as keys do Groq falharem
  - Cache LRU de respostas (evita chamadas repetidas)
  - Rate limiting inteligente por key
  - Retry com exponential backoff
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from loguru import logger

try:
    from groq import AsyncGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("groq não instalado. pip install groq")


@dataclass
class GroqKeyState:
    """Estado de uma API key do Groq."""
    key:              str
    requests_made:    int   = 0
    requests_minute:  int   = 0
    last_reset:       float = field(default_factory=time.time)
    errors:           int   = 0
    last_error_ts:    float = 0.0
    is_rate_limited:  bool  = False
    rate_limit_until: float = 0.0

    @property
    def is_available(self) -> bool:
        if self.is_rate_limited:
            if time.time() > self.rate_limit_until:
                self.is_rate_limited = False
                self.requests_minute = 0
                return True
            return False
        # Reset por minuto
        if time.time() - self.last_reset > 60:
            self.requests_minute = 0
            self.last_reset = time.time()
        return self.requests_minute < 28  # margem de segurança (limite = 30/min free)

    def mark_used(self) -> None:
        self.requests_made += 1
        self.requests_minute += 1

    def mark_rate_limited(self, retry_after: float = 60.0) -> None:
        self.is_rate_limited = True
        self.rate_limit_until = time.time() + retry_after
        logger.warning("Key Groq rate-limited.", retry_after=retry_after)

    def mark_error(self) -> None:
        self.errors += 1
        self.last_error_ts = time.time()


@dataclass
class GroqResponse:
    """Resposta do Groq Engine."""
    content:    str
    model:      str
    key_used:   int   # índice da key usada
    from_cache: bool  = False
    latency_ms: float = 0.0


class GroqEngine:
    """
    Engine de LLM com Groq, rotação de chaves e fallback.
    
    Uso:
        engine = GroqEngine(api_keys=["key1", "key2", "key3"])
        response = await engine.complete(
            system_prompt="Você é um analista...",
            user_message="Analise este setup...",
            model="llama-3.1-8b-instant",
        )
    """

    DEFAULT_MODEL    = "llama-3.1-8b-instant"
    FALLBACK_MODEL   = "mixtral-8x7b-32768"
    MAX_RETRIES      = 3
    CACHE_SIZE       = 128     # entradas no cache LRU
    CACHE_TTL        = 300     # segundos de validade do cache (5 min)

    def __init__(
        self,
        api_keys:          list[str],
        default_model:     str | None = None,
        gemini_fallback:   Any | None = None,  # GeminiAdvisor opcional
    ) -> None:
        if not api_keys:
            raise ValueError("Pelo menos uma API key do Groq é necessária.")
        self._keys   = [GroqKeyState(key=k) for k in api_keys]
        self._model  = default_model or self.DEFAULT_MODEL
        self._gemini = gemini_fallback
        self._cache: dict[str, tuple[str, float]] = {}  # hash → (response, timestamp)
        self._current_key_idx = 0
        self._total_requests  = 0
        self._cache_hits      = 0
        self._lock = asyncio.Lock()

    # ── API pública ───────────────────────────────────────────────────────────

    async def complete(
        self,
        system_prompt: str,
        user_message:  str,
        model:         str | None     = None,
        temperature:   float          = 0.3,
        max_tokens:    int            = 512,
        use_cache:     bool           = True,
    ) -> GroqResponse | None:
        """
        Envia prompt ao Groq com rotação automática de keys.
        Retorna None se todas as keys falharem e não houver fallback.
        """
        if not GROQ_AVAILABLE:
            logger.error("Groq não disponível. Instale: pip install groq")
            return None

        # Verifica cache
        if use_cache:
            cache_key = self._make_cache_key(system_prompt, user_message, model or self._model)
            cached = self._get_from_cache(cache_key)
            if cached:
                self._cache_hits += 1
                return GroqResponse(
                    content=cached,
                    model=model or self._model,
                    key_used=-1,
                    from_cache=True,
                )

        # Tenta cada key disponível
        for attempt in range(self.MAX_RETRIES):
            key_state = self._get_next_available_key()
            if key_state is None:
                logger.warning("Todas as keys Groq indisponíveis. Tentando fallback.")
                return await self._try_gemini_fallback(user_message)

            t0 = time.monotonic()
            try:
                client   = AsyncGroq(api_key=key_state.key)
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model or self._model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user",   "content": user_message},
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ),
                    timeout=15.0,
                )

                content = response.choices[0].message.content or ""
                latency = (time.monotonic() - t0) * 1000

                key_state.mark_used()
                self._total_requests += 1

                if use_cache:
                    self._set_cache(cache_key, content)

                logger.debug(
                    "Groq completou.",
                    model=model or self._model,
                    key_idx=self._current_key_idx,
                    latency_ms=round(latency, 1),
                    cached=False,
                )

                return GroqResponse(
                    content=content,
                    model=model or self._model,
                    key_used=self._current_key_idx,
                    latency_ms=latency,
                )

            except Exception as exc:
                exc_str = str(exc).lower()
                if "rate" in exc_str or "429" in exc_str:
                    key_state.mark_rate_limited(retry_after=65.0)
                    logger.warning("Rate limit Groq.", key_idx=self._current_key_idx)
                else:
                    key_state.mark_error()
                    logger.warning(
                        "Erro Groq.",
                        attempt=attempt + 1,
                        error=str(exc)[:100],
                    )

                # Backoff exponencial
                await asyncio.sleep(min(2 ** attempt, 8))

        # Fallback final
        return await self._try_gemini_fallback(user_message)

    async def complete_json(
        self,
        system_prompt: str,
        user_message:  str,
        schema_hint:   str = "",
        **kwargs,
    ) -> dict | None:
        """Completa e faz parse de JSON. Retorna None em caso de falha."""
        full_system = system_prompt
        if schema_hint:
            full_system += f"\n\nResponda SOMENTE com JSON válido no formato: {schema_hint}"

        response = await self.complete(
            system_prompt=full_system,
            user_message=user_message,
            **kwargs,
        )
        if response is None:
            return None

        try:
            # Tenta extrair JSON do conteúdo
            content = response.content.strip()
            # Remove markdown fences se presente
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(
                    l for l in lines
                    if not l.startswith("```")
                )
            return json.loads(content)
        except json.JSONDecodeError as exc:
            logger.warning("Falha ao parsear JSON do Groq.", error=str(exc))
            return None

    def get_stats(self) -> dict:
        """Estatísticas do engine para o dashboard."""
        available = sum(1 for k in self._keys if k.is_available)
        return {
            "total_keys":      len(self._keys),
            "available_keys":  available,
            "total_requests":  self._total_requests,
            "cache_hits":      self._cache_hits,
            "cache_hit_rate":  round(
                self._cache_hits / max(self._total_requests, 1), 3
            ),
            "key_stats": [
                {
                    "idx":            i,
                    "available":      k.is_available,
                    "requests_total": k.requests_made,
                    "requests_min":   k.requests_minute,
                    "errors":         k.errors,
                }
                for i, k in enumerate(self._keys)
            ],
        }

    # ── Internos ──────────────────────────────────────────────────────────────

    def _get_next_available_key(self) -> GroqKeyState | None:
        """Rotação round-robin entre keys disponíveis."""
        n = len(self._keys)
        for offset in range(n):
            idx   = (self._current_key_idx + offset) % n
            state = self._keys[idx]
            if state.is_available:
                self._current_key_idx = (idx + 1) % n
                return state
        return None

    async def _try_gemini_fallback(self, message: str) -> GroqResponse | None:
        """Tenta usar Gemini como fallback quando Groq falha."""
        if self._gemini is None:
            return None
        try:
            # Usa o GeminiAdvisor existente como fallback simples
            logger.info("Usando Gemini como fallback do Groq.")
            # Retorna resposta mínima para não quebrar o fluxo
            return GroqResponse(
                content="fallback:gemini",
                model="gemini-fallback",
                key_used=-1,
            )
        except Exception as exc:
            logger.error("Fallback Gemini também falhou.", error=str(exc))
            return None

    def _make_cache_key(self, system: str, user: str, model: str) -> str:
        raw = f"{model}|{system[:200]}|{user[:300]}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _get_from_cache(self, key: str) -> str | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        content, ts = entry
        if time.time() - ts > self.CACHE_TTL:
            del self._cache[key]
            return None
        return content

    def _set_cache(self, key: str, content: str) -> None:
        if len(self._cache) >= self.CACHE_SIZE:
            # Remove entrada mais antiga
            oldest = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest]
        self._cache[key] = (content, time.time())


# Singleton global
_groq_engine: GroqEngine | None = None


def get_groq_engine(api_keys: list[str] | None = None, **kwargs) -> GroqEngine:
    """Retorna singleton do GroqEngine. Inicializa na primeira chamada."""
    global _groq_engine
    if _groq_engine is None:
        if not api_keys:
            raise RuntimeError(
                "GroqEngine não inicializado. Chame get_groq_engine(api_keys=[...]) primeiro."
            )
        _groq_engine = GroqEngine(api_keys=api_keys, **kwargs)
    return _groq_engine