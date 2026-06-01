from __future__ import annotations

import time
from collections import OrderedDict
from threading import Lock
from typing import Any


class LRUCache:
    """
    Cache LRU thread-safe de uso geral.

    Implementação própria usando OrderedDict para evitar
    dependência externa e ter controle total sobre o comportamento.
    """

    def __init__(self, maxsize: int = 1024) -> None:
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._maxsize = maxsize
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def delete(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def keys(self) -> list[str]:
        """Retorna snapshot das chaves atuais (thread-safe)."""
        with self._lock:
            return list(self._cache.keys())

    @property
    def stats(self) -> dict[str, int | float]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
                "size": len(self._cache),
                "maxsize": self._maxsize,
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: str) -> bool:
        # FIX: Adicionado lock para evitar race condition de leitura
        # concorrente com set/delete em outra thread.
        with self._lock:
            return key in self._cache


class FeatureCache:
    """
    Cache singleton de features de mercado por símbolo.

    Cada símbolo tem seu próprio namespace para evitar colisão.
    Usado pelos modelos ML e pelo ExecutionEngine para evitar
    re-computar features a cada tick.

    Suporta invalidação por TTL via is_expired(), usado pelo
    ExecutionEngine para detectar cache stale pós-reconexão.
    """

    _instance: "FeatureCache | None" = None
    _instance_lock: Lock = Lock()

    def __init__(self, maxsize: int = 2048) -> None:
        self._cache = LRUCache(maxsize=maxsize)
        # FIX: Timestamps por chave para suportar is_expired().
        # Usa time.monotonic() — imune a ajustes de relógio do sistema.
        self._timestamps: dict[str, float] = {}
        self._ts_lock = Lock()

    @classmethod
    def initialize(cls, maxsize: int = 2048) -> "FeatureCache":
        """Inicializa o singleton (idempotente)."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(maxsize=maxsize)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "FeatureCache":
        """
        Retorna a instância singleton.

        FIX: Ao invés de lançar RuntimeError (comportamento frágil em testes),
        inicializa com defaults se ainda não existir.
        Chamadores que precisam de maxsize customizado devem usar initialize().
        """
        if cls._instance is None:
            return cls.initialize()
        return cls._instance

    def set_features(self, symbol: str, key: str, value: Any) -> None:
        """
        Armazena features e registra timestamp para controle de TTL.
        """
        cache_key = f"{symbol}:{key}"
        self._cache.set(cache_key, value)
        # FIX: Registra timestamp após cada set para que is_expired() funcione.
        with self._ts_lock:
            self._timestamps[cache_key] = time.monotonic()

    def get_features(self, symbol: str, key: str) -> Any | None:
        return self._cache.get(f"{symbol}:{key}")

    def is_expired(self, symbol: str, key: str, ttl_seconds: float) -> bool:
        """
        Retorna True se o entry de cache é mais antigo que ttl_seconds.

        Usado pelo ExecutionEngine para detectar cache stale após
        reconexão WebSocket ou pausa longa do event loop.

        Args:
            symbol: Símbolo de mercado (ex: "R_50").
            key: Chave do entry (ex: "features_df").
            ttl_seconds: Tempo máximo de vida em segundos.

        Returns:
            True se expirado ou nunca setado; False se ainda válido.
        """
        cache_key = f"{symbol}:{key}"
        with self._ts_lock:
            ts = self._timestamps.get(cache_key)
        if ts is None:
            return True  # Nunca setado → tratar como expirado.
        return (time.monotonic() - ts) > ttl_seconds

    def invalidate_symbol(self, symbol: str) -> None:
        """
        Remove todas as features de um símbolo do cache.

        FIX: Usa self._cache.keys() (método público thread-safe) ao invés
        de acessar self._cache._cache diretamente (violação de encapsulamento
        que também não era thread-safe).
        """
        prefix = f"{symbol}:"
        keys_to_delete = [k for k in self._cache.keys() if k.startswith(prefix)]
        for k in keys_to_delete:
            self._cache.delete(k)
        # Remove timestamps associados.
        with self._ts_lock:
            for k in keys_to_delete:
                self._timestamps.pop(k, None)

    def invalidate_key(self, symbol: str, key: str) -> None:
        """Remove uma feature específica de um símbolo."""
        cache_key = f"{symbol}:{key}"
        self._cache.delete(cache_key)
        with self._ts_lock:
            self._timestamps.pop(cache_key, None)

    def clear_all(self) -> None:
        """Limpa todo o cache e timestamps."""
        self._cache.clear()
        with self._ts_lock:
            self._timestamps.clear()

    @property
    def stats(self) -> dict:
        base = self._cache.stats
        with self._ts_lock:
            base["tracked_timestamps"] = len(self._timestamps)
        return base