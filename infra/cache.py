from __future__ import annotations

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

    @property
    def stats(self) -> dict[str, int | float]:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total else 0.0,
            "size": len(self._cache),
            "maxsize": self._maxsize,
        }

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        return key in self._cache


class FeatureCache:
    """
    Cache singleton de features de mercado por símbolo.

    Cada símbolo tem seu próprio namespace para evitar colisão.
    Usado pelos modelos ML para evitar re-computar features
    a cada tick.
    """

    _instance: "FeatureCache | None" = None
    _cache: LRUCache

    def __init__(self, maxsize: int = 2048) -> None:
        self._cache = LRUCache(maxsize=maxsize)

    @classmethod
    def initialize(cls, maxsize: int = 2048) -> "FeatureCache":
        if cls._instance is None:
            cls._instance = cls(maxsize=maxsize)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "FeatureCache":
        if cls._instance is None:
            raise RuntimeError("FeatureCache não inicializado. Chame initialize() primeiro.")
        return cls._instance

    def set_features(self, symbol: str, key: str, value: Any) -> None:
        self._cache.set(f"{symbol}:{key}", value)

    def get_features(self, symbol: str, key: str) -> Any | None:
        return self._cache.get(f"{symbol}:{key}")

    def invalidate_symbol(self, symbol: str) -> None:
        """Remove todas as features de um símbolo do cache."""
        keys_to_delete = [k for k in self._cache._cache if k.startswith(f"{symbol}:")]
        for k in keys_to_delete:
            self._cache.delete(k)

    @property
    def stats(self) -> dict:
        return self._cache.stats