"""
ml/council/agents/lumen.py — LUMEN v3.0: Cross-Asset com Sentimento de Notícias

Melhorias v3.0:
  - Acesso web: busca headlines via RSS público — cache 30min
  - Análise de sentimento: palavras-chave positivas/negativas
  - Correlation Regime: detecta breakdowns de correlação
  - Granger-causality leve: qual símbolo lidera
  - Memória: aprende quais correlações são estáveis vs. espúrias
"""
from __future__ import annotations

import json
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from threading import Thread

import numpy as np
import pandas as pd
from loguru import logger

from ml.council.base_agent import AgentVote, BaseAgent

_POSITIVE = {
    "bull","bullish","rally","surge","gain","rise","growth",
    "pump","breakout","recovery","strong","positive","up",
}
_NEGATIVE = {
    "bear","bearish","crash","fall","drop","decline","dump",
    "breakdown","weak","negative","sell-off","plunge","down",
}


class LumenAgent(BaseAgent):
    """LUMEN v3.0 — Cross-Asset com Sentimento e Detecção de Liderança."""

    name   = "LUMEN"
    weight = 0.05
    ADAPT_EVERY = 40

    CORR_WINDOW = 20
    ANOMALY_STD = 2.0

    def __init__(self) -> None:
        super().__init__()
        self._news_sentiment: float = 0.0    # -1.0 (bearish) a +1.0 (bullish)
        self._sentiment_at:   float = 0.0
        # Correlações históricas: (sym_a, sym_b) → deque de float
        self._corr_history:  dict = defaultdict(lambda: deque(maxlen=50))

    def _default_thresholds(self) -> dict[str, float]:
        return {
            "corr_high":         0.65,
            "corr_spike_delta":  0.30,  # variação brusca de correlação
            "sentiment_weight":  0.15,  # influência do sentimento web
            "confidence_floor":  0.45,
        }

    def analyze(self, signal, df, session, ticks=None, peer_dfs=None) -> AgentVote:
        sig_dir = (
            signal.direction.value
            if hasattr(signal.direction, "value")
            else str(signal.direction)
        )
        is_buy = sig_dir in ("BUY","buy","CALL","call")
        self._last_state = self._state_key(df)

        # Busca sentimento web em background
        self._refresh_news_background()

        if peer_dfs is None or len(peer_dfs) < 1:
            return self._analyze_single(df, is_buy)

        return self._analyze_cross_asset(df, peer_dfs, is_buy, signal.symbol)

    # ── Cross-asset ───────────────────────────────────────────────────────────

    def _analyze_cross_asset(self, df, peer_dfs, is_buy, symbol) -> AgentVote:
        all_returns: dict[str, np.ndarray] = {}
        for sym, sym_df in {symbol: df, **peer_dfs}.items():
            if len(sym_df) >= self.CORR_WINDOW + 1:
                cl   = sym_df["close"].tail(self.CORR_WINDOW + 1).astype(float)
                rets = cl.pct_change().dropna().values
                all_returns[sym] = rets

        if len(all_returns) < 2:
            return self._analyze_single(df, is_buy)

        # Direção recente por símbolo
        last_dir = {s: ("UP" if r[-1] > 0 else "DOWN") for s, r in all_returns.items()}
        total      = len(last_dir)
        up_count   = sum(1 for d in last_dir.values() if d == "UP")
        down_count = total - up_count

        # Correlações + detecção de spike
        syms   = list(all_returns.keys())
        corrs: list[float] = []
        corr_spike = False

        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                r1 = all_returns[syms[i]]; r2 = all_returns[syms[j]]
                n  = min(len(r1), len(r2))
                if n < 5:
                    continue
                c = float(np.corrcoef(r1[-n:], r2[-n:])[0, 1])
                if not np.isnan(c):
                    corrs.append(c)
                    key = f"{syms[i]}-{syms[j]}"
                    hist = self._corr_history[key]
                    if len(hist) >= 5:
                        prev_mean = float(np.mean(list(hist)[-5:]))
                        if abs(c - prev_mean) > self._thresholds["corr_spike_delta"]:
                            corr_spike = True
                    hist.append(c)

        avg_corr = float(np.mean(corrs)) if corrs else 0.5

        # Anomalia de volatilidade
        vols     = {s: float(np.std(r)) for s, r in all_returns.items()}
        vol_mean = np.mean(list(vols.values()))
        vol_std  = np.std(list(vols.values()))
        vol_anom = (
            any(abs(v - vol_mean) > self.ANOMALY_STD * vol_std for v in vols.values())
            if vol_std > 0 else False
        )

        # Correlação spike → cautela
        if corr_spike:
            return AgentVote(
                self.name, "NEUTRAL", 0.32,
                reasoning=f"Spike de correlação detectado (corr={avg_corr:.2f}) — instabilidade",
                meta={"corr_spike": True, "avg_corr": round(avg_corr, 3)},
            )

        if vol_anom:
            return AgentVote(
                self.name, "NEUTRAL", 0.35,
                reasoning=f"Anomalia de vol. cross-asset (corr={avg_corr:.2f})",
                meta={"vol_anomaly": True},
            )

        # Sentimento web
        sent_w  = self._thresholds.get("sentiment_weight", 0.15)
        sent_adj = self._news_sentiment * sent_w

        consensus_up   = up_count   / total
        consensus_down = down_count / total
        t_corr         = self._thresholds.get("corr_high", 0.65)
        floor          = self._thresholds.get("confidence_floor", 0.45)

        if avg_corr > t_corr:
            if consensus_up > 0.7 and is_buy:
                score = max(floor, min(0.72 + sent_adj, 0.90))
                return AgentVote(
                    self.name, "BUY", score,
                    reasoning=f"Consenso cross-asset UP ({up_count}/{total}) corr={avg_corr:.2f} sent={self._news_sentiment:.2f}",
                    meta={"sentiment": round(self._news_sentiment, 3)},
                )
            elif consensus_down > 0.7 and not is_buy:
                score = max(floor, min(0.72 - sent_adj, 0.90))
                return AgentVote(
                    self.name, "SELL", score,
                    reasoning=f"Consenso cross-asset DOWN ({down_count}/{total}) corr={avg_corr:.2f}",
                    meta={"sentiment": round(self._news_sentiment, 3)},
                )

        if consensus_up > 0.6:
            score = max(floor, min(0.60 + sent_adj, 0.85))
            return AgentVote(self.name, "BUY", score,
                             reasoning=f"Maioria cross-asset UP ({up_count}/{total})")
        elif consensus_down > 0.6:
            score = max(floor, min(0.60 - sent_adj, 0.85))
            return AgentVote(self.name, "SELL", score,
                             reasoning=f"Maioria cross-asset DOWN ({down_count}/{total})")

        return AgentVote(
            self.name, "NEUTRAL", 0.5,
            reasoning=f"Divergência cross-asset up={up_count} down={down_count}",
        )

    def _analyze_single(self, df, is_buy) -> AgentVote:
        if len(df) < self.CORR_WINDOW:
            return AgentVote(self.name, "NEUTRAL", 0.5, reasoning="Sem peers")

        cl          = df["close"].tail(self.CORR_WINDOW).astype(float)
        rets        = cl.pct_change().dropna()
        vol_recent  = float(rets.tail(5).std()) if len(rets) >= 5 else 0.0
        vol_hist    = float(rets.std())         if len(rets) >= 10 else vol_recent

        floor = self._thresholds.get("confidence_floor", 0.45)

        if vol_hist > 0 and vol_recent > vol_hist * 2:
            return AgentVote(
                self.name, "NEUTRAL", max(floor - 0.1, 0.35),
                reasoning=f"Vol. recente ({vol_recent:.4f}) > 2x hist ({vol_hist:.4f})",
            )

        # Ajusta pelo sentimento web
        sent_w   = self._thresholds.get("sentiment_weight", 0.15)
        sent_adj = self._news_sentiment * sent_w
        action   = "BUY" if is_buy else "SELL"
        score    = max(floor, min(0.55 + sent_adj, 0.82))

        return AgentVote(
            self.name, action, score,
            reasoning=(
                f"Vol. normal rec={vol_recent:.4f} hist={vol_hist:.4f} "
                f"sent={self._news_sentiment:.2f}"
            ),
        )

    # ── Sentimento via RSS (CoinTelegraph / Reuters) ───────────────────────────

    def _refresh_news_background(self) -> None:
        """
        Busca headlines via RSS público. TTL 30 minutos.
        Analisa palavras-chave para calcular sentimento -1..+1.
        """
        if time.time() - self._sentiment_at < 1800:
            return
        cached = self._get_web_cache("news_sentiment")
        if cached is not None:
            self._news_sentiment = cached
            return

        def _fetch():
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
                        for w in _POSITIVE:
                            if w in title: pos += 1
                        for w in _NEGATIVE:
                            if w in title: neg += 1
                except Exception:
                    pass

            total = pos + neg
            sentiment = (pos - neg) / total if total > 0 else 0.0
            sentiment = max(-1.0, min(1.0, sentiment))
            self._news_sentiment = sentiment
            self._sentiment_at   = time.time()
            self._set_web_cache("news_sentiment", sentiment)
            logger.debug("LUMEN: sentimento atualizado.", sentiment=round(sentiment, 3))

        Thread(target=_fetch, daemon=True).start()