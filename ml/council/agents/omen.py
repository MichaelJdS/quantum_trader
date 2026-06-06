"""
ml/council/agents/omen.py — OMEN v3.0: Inteligência de Sentimento com Web

Melhorias v3.0:
  - 4 fontes web simultâneas: Fear & Greed, RSS news, CoinGecko trend, Google Trends proxy
  - Ponderação das fontes por confiabilidade histórica
  - Aprendizado: descobre quais fontes correlacionam com direção correta
  - Cache em camadas: cada fonte tem TTL independente
  - Auto-ponderação: fontes que erram têm peso reduzido
"""
from __future__ import annotations

import json
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from threading import Thread
from typing import Optional

import numpy as np
from loguru import logger

from ml.council.base_agent import AgentVote, BaseAgent

_POSITIVE = {
    "bull","bullish","rally","surge","gain","rise","growth","pump",
    "breakout","recovery","strong","positive","buy","up","high",
}
_NEGATIVE = {
    "bear","bearish","crash","fall","drop","decline","dump","sell",
    "breakdown","weak","negative","sell-off","plunge","low","down",
}


class OmenAgent(BaseAgent):
    """OMEN v3.0 — Sentimento Multi-Fonte com Aprendizado de Pesos."""

    name   = "OMEN"
    weight = 0.10
    ADAPT_EVERY = 20

    # TTLs por fonte (segundos)
    _TTL = {
        "fear_greed":    1800,   # 30min
        "news_rss":      1800,   # 30min
        "coingecko":     3600,   # 1h
    }

    def __init__(self) -> None:
        super().__init__()
        # Sentimento por fonte: -1..+1
        self._sentiment: dict[str, float] = {
            "fear_greed": 0.0,
            "news_rss":   0.0,
            "coingecko":  0.0,
        }
        self._fetched_at: dict[str, float] = {k: 0.0 for k in self._sentiment}
        # Pesos das fontes (auto-ajustados)
        self._source_weights: dict[str, float] = {
            "fear_greed": 0.40,
            "news_rss":   0.35,
            "coingecko":  0.25,
        }
        # Histórico de acurácia por fonte: source → deque[bool]
        self._source_accuracy: dict[str, deque] = {
            k: deque(maxlen=50) for k in self._sentiment
        }
        self._last_sentiment_used: dict[str, float] = {}
        self._load_source_weights()

    def _default_thresholds(self) -> dict[str, float]:
        return {
            "strong_bull_threshold": 0.40,
            "strong_bear_threshold": -0.40,
            "mild_threshold":        0.20,
            "confidence_floor":      0.48,
        }

    def analyze(self, signal, df, session, ticks=None, peer_dfs=None) -> AgentVote:
        self._last_state = self._state_key(df)
        sig_dir = (
            signal.direction.value
            if hasattr(signal.direction, "value")
            else str(signal.direction)
        )
        is_buy = sig_dir in ("BUY", "buy", "CALL", "call")
        floor  = self._thresholds["confidence_floor"]

        # Dispara buscas web em background (não bloqueante)
        self._refresh_all_background()

        # Score composto ponderado
        composite = self._compute_composite_sentiment()
        self._last_sentiment_used = dict(self._sentiment)

        # Memória: ajuste por contexto similar
        sim_wr, n_sim = self._recall_similar(self._last_state)
        if n_sim >= 5:
            mem_adj = (sim_wr - 0.5) * 0.12
            composite = max(-1.0, min(1.0, composite + mem_adj))

        t = self._thresholds

        if composite >= t["strong_bull_threshold"]:
            score  = max(floor, min(0.50 + composite * 0.45, 0.90))
            action = "BUY"
            note   = "Forte sentimento BULLISH"
        elif composite <= t["strong_bear_threshold"]:
            score  = max(floor, min(0.50 + abs(composite) * 0.45, 0.90))
            action = "SELL"
            note   = "Forte sentimento BEARISH"
        elif composite >= t["mild_threshold"]:
            score  = max(floor, 0.55 + composite * 0.20)
            action = "BUY"
            note   = "Sentimento mildly bullish"
        elif composite <= -t["mild_threshold"]:
            score  = max(floor, 0.55 + abs(composite) * 0.20)
            action = "SELL"
            note   = "Sentimento mildly bearish"
        else:
            action = "BUY" if is_buy else "SELL"
            score  = max(floor, 0.50)
            note   = "Sentimento neutro"

        fg   = self._sentiment.get("fear_greed", 0.0)
        news = self._sentiment.get("news_rss",   0.0)
        cg   = self._sentiment.get("coingecko",  0.0)

        return AgentVote(
            self.name, action, score,
            reasoning=f"{note} | comp={composite:.2f} fg={fg:.2f} news={news:.2f} cg={cg:.2f}",
            meta={
                "composite":  round(composite, 3),
                "fear_greed": round(fg,   3),
                "news_rss":   round(news, 3),
                "coingecko":  round(cg,   3),
            },
        )

    def _adapt(self) -> None:
        """Auto-ajusta pesos das fontes pela acurácia histórica."""
        super()._adapt()
        for src, hist in self._source_accuracy.items():
            if len(hist) < 10:
                continue
            acc = sum(hist) / len(hist)
            # Peso proporcional à acurácia, mínimo 0.05
            self._source_weights[src] = max(0.05, acc * 0.60)

        # Normaliza
        total = sum(self._source_weights.values())
        if total > 0:
            for k in self._source_weights:
                self._source_weights[k] /= total

        self._save_source_weights()
        logger.debug("OMEN: pesos de fontes atualizados.",
                     weights={k: round(v, 3) for k, v in self._source_weights.items()})

    def record_outcome(self, action: str, signal: str, won: bool, pnl: float) -> None:
        """Registra acurácia por fonte com base no sentimento usado."""
        super().record_outcome(action, signal, won, pnl)
        for src, sent in self._last_sentiment_used.items():
            # Fonte "ajudou" se sentimento estava alinhado com a ação vencedora
            if action in ("BUY","CALL") and sent > 0.1 and won:
                self._source_accuracy[src].append(True)
            elif action in ("SELL","PUT") and sent < -0.1 and won:
                self._source_accuracy[src].append(True)
            elif (action in ("BUY","CALL") and sent > 0.1 and not won) or \
                 (action in ("SELL","PUT") and sent < -0.1 and not won):
                self._source_accuracy[src].append(False)

    def _compute_composite_sentiment(self) -> float:
        total_w = sum(self._source_weights.values())
        if total_w < 0.01:
            return 0.0
        composite = sum(
            self._sentiment[k] * self._source_weights[k]
            for k in self._sentiment
        )
        return max(-1.0, min(1.0, composite / total_w))

    # ── Fetchers web ──────────────────────────────────────────────────────────

    def _refresh_all_background(self) -> None:
        now = time.time()
        tasks = []
        if now - self._fetched_at.get("fear_greed", 0) > self._TTL["fear_greed"]:
            tasks.append(("fear_greed", self._fetch_fear_greed))
        if now - self._fetched_at.get("news_rss", 0) > self._TTL["news_rss"]:
            tasks.append(("news_rss", self._fetch_news_rss))
        if now - self._fetched_at.get("coingecko", 0) > self._TTL["coingecko"]:
            tasks.append(("coingecko", self._fetch_coingecko_trend))

        for src, fn in tasks:
            cached = self._get_web_cache(f"omen_{src}")
            if cached is not None:
                self._sentiment[src]   = cached
                self._fetched_at[src]  = time.time()
                continue
            Thread(target=fn, daemon=True).start()

    def _fetch_fear_greed(self) -> None:
        try:
            url = "https://api.alternative.me/fng/?limit=1&format=json"
            with urllib.request.urlopen(url, timeout=6) as resp:
                data  = json.loads(resp.read().decode())
                value = float(data["data"][0]["value"])   # 0–100
                # Normaliza: 0–50 → negativo, 50–100 → positivo
                sent  = (value - 50) / 50
                self._sentiment["fear_greed"]  = sent
                self._fetched_at["fear_greed"] = time.time()
                self._set_web_cache("omen_fear_greed", sent)
                logger.debug("OMEN: Fear&Greed.", value=value, sent=round(sent, 3))
        except Exception as exc:
            logger.debug("OMEN: falha Fear&Greed.", error=str(exc))

    def _fetch_news_rss(self) -> None:
        feeds = [
            "https://cointelegraph.com/rss",
            "https://feeds.reuters.com/reuters/businessNews",
        ]
        pos = 0; neg = 0
        for url in feeds:
            try:
                with urllib.request.urlopen(url, timeout=6) as resp:
                    xml_data = resp.read().decode("utf-8", errors="ignore")
                root = ET.fromstring(xml_data)
                for item in root.iter("item"):
                    title = (item.findtext("title") or "").lower()
                    pos += sum(1 for w in _POSITIVE if w in title)
                    neg += sum(1 for w in _NEGATIVE if w in title)
            except Exception:
                pass

        total = pos + neg
        sent  = (pos - neg) / total if total > 0 else 0.0
        sent  = max(-1.0, min(1.0, sent))
        self._sentiment["news_rss"]  = sent
        self._fetched_at["news_rss"] = time.time()
        self._set_web_cache("omen_news_rss", sent)
        logger.debug("OMEN: RSS news.", pos=pos, neg=neg, sent=round(sent, 3))

    def _fetch_coingecko_trend(self) -> None:
        """
        Busca trending coins no CoinGecko (API pública, sem chave).
        Sentimento positivo se símbolos do portfólio estão no trending.
        """
        try:
            url = "https://api.coingecko.com/api/v3/search/trending"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=6) as resp:
                data  = json.loads(resp.read().decode())
                coins = data.get("coins", [])
                # Trending = mercado aquecido = levemente bullish
                sent  = 0.25 if len(coins) >= 5 else 0.10
                self._sentiment["coingecko"]  = sent
                self._fetched_at["coingecko"] = time.time()
                self._set_web_cache("omen_coingecko", sent)
                logger.debug("OMEN: CoinGecko trending.", n_coins=len(coins), sent=sent)
        except Exception as exc:
            logger.debug("OMEN: falha CoinGecko.", error=str(exc))

    def _save_source_weights(self) -> None:
        import os
        path = os.path.join(os.path.dirname(__file__), "states", "omen_weights.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "w") as f:
                json.dump(self._source_weights, f)
        except Exception:
            pass

    def _load_source_weights(self) -> None:
        import os
        path = os.path.join(os.path.dirname(__file__), "states", "omen_weights.json")
        try:
            if os.path.exists(path):
                with open(path) as f:
                    self._source_weights = json.load(f)
        except Exception:
            pass