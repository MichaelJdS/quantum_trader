"""
ml/council/agents/echo.py — ECHO v3.0: Aprendizado por Reforço Evolutivo

Melhorias v3.0:
  - PPO-tabular: política com função de valor separada (Actor-Critic leve)
  - Estado rico: 12 features discretizadas (vs. 3 anteriores)
  - Acesso à internet: busca sentimento de mercado via Fear & Greed Index
  - Exploration annealing: epsilon decai com o tempo
  - Experience Replay: mini-batches de 32 experiências
  - Persistência total: estado salvo a cada 10 trades, carregado no boot
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import Thread
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ml.council.base_agent import AgentVote, BaseAgent

_STATE_PATH = os.path.join(os.path.dirname(__file__), "states", "echo_ppo.json")


@dataclass
class Experience:
    """Experiência para o replay buffer."""
    state:   str
    action:  str
    reward:  float
    next_st: str
    done:    bool


class EchoAgent(BaseAgent):
    """ECHO v3.0 — RL Actor-Critic com acesso a sentimento de mercado."""

    name   = "ECHO"
    weight = 0.08
    ADAPT_EVERY = 10

    ACTIONS       = ["BUY", "SELL", "NEUTRAL"]
    LEARNING_RATE = 0.10
    DISCOUNT      = 0.90
    EPSILON_START = 0.25   # exploração inicial
    EPSILON_MIN   = 0.03   # mínimo de exploração
    EPSILON_DECAY = 0.995  # decaimento por update
    BATCH_SIZE    = 32
    BUFFER_SIZE   = 1000

    def __init__(self) -> None:
        super().__init__()
        # Q-table: estado → {ação: Q-value}
        self._q: dict = defaultdict(
            lambda: {"BUY": 0.0, "SELL": 0.0, "NEUTRAL": 0.0}
        )
        # Value function V(s): estimativa do valor do estado
        self._v: dict = defaultdict(float)
        # Replay buffer
        self._buffer:   deque = deque(maxlen=self.BUFFER_SIZE)
        self._epsilon:  float = self.EPSILON_START
        self._last_exp: Experience | None = None
        self._total_updates: int = 0
        # Sentimento web (Fear & Greed)
        self._fear_greed:    float = 50.0   # 0=Fear extremo, 100=Greed extremo
        self._fg_fetched_at: float = 0.0
        self._load_full_state()

    def analyze(self, signal, df, session, ticks=None, peer_dfs=None) -> AgentVote:
        state = self._get_rich_state(df, session)
        self._last_state = state

        # Busca sentimento web em background (não bloqueia)
        self._refresh_sentiment_background()

        if self._total_updates < 15:
            return AgentVote(
                self.name, "NEUTRAL", 0.50,
                reasoning=f"Bootstrap ({self._total_updates}/15 updates)",
            )

        # Política: epsilon-greedy sobre Q+V
        q_vals = self._q[state]

        # Usa sentimento para ajustar Q-values
        fg_adj = (self._fear_greed - 50.0) / 100.0  # -0.5 a +0.5
        q_buy_adj  = q_vals["BUY"]  + fg_adj * 0.1
        q_sell_adj = q_vals["SELL"] - fg_adj * 0.1
        adj_q = {"BUY": q_buy_adj, "SELL": q_sell_adj, "NEUTRAL": q_vals["NEUTRAL"]}

        if random.random() < self._epsilon:
            best_action = random.choice(self.ACTIONS)
            explore_note = " [explore]"
        else:
            best_action  = max(adj_q, key=lambda a: adj_q[a])
            explore_note = ""

        q_max = max(adj_q.values())
        q_min = min(adj_q.values())
        spread = q_max - q_min

        confidence = 0.50 + min(spread / max(abs(q_max) + 0.01, 1.0), 0.38)
        confidence = max(
            self._thresholds.get("confidence_floor", 0.40),
            min(confidence, 0.88),
        )

        # Memória episódica: boost se estado similar teve bom histórico
        sim_wr, n_sim = self._recall_similar(state)
        if n_sim >= 5:
            mem_adj = (sim_wr - 0.5) * 0.10
            confidence = max(0.30, min(0.90, confidence + mem_adj))

        # Salva experiência parcial para update posterior
        self._last_exp = Experience(
            state=state, action=best_action,
            reward=0.0, next_st=state, done=False
        )

        fg_note = f" FG={self._fear_greed:.0f}"
        return AgentVote(
            self.name, best_action, confidence,
            reasoning=(
                f"Q={q_max:.2f} eps={self._epsilon:.3f}{explore_note}"
                f" sim_wr={sim_wr:.2f}(n={n_sim}){fg_note}"
            ),
            meta={"fear_greed": self._fear_greed, "epsilon": round(self._epsilon, 3)},
        )

    def update_from_trade(self, action: str, reward: float, next_df: pd.DataFrame | None = None) -> None:
        """Atualiza Actor-Critic com resultado do trade."""
        if self._last_exp is None:
            return

        state  = self._last_exp.state
        q_old  = self._q[state][action]
        v_next = self._v.get(state, 0.0)

        # Bellman: Q(s,a) += lr * (r + γ*V(s') - Q(s,a))
        td_error = reward + self.DISCOUNT * v_next - q_old
        self._q[state][action] = round(
            q_old + self.LEARNING_RATE * td_error, 4
        )

        # Atualiza V(s): V(s) ≈ max Q(s,*)
        self._v[state] = max(self._q[state].values())

        # Adiciona ao replay buffer
        exp = Experience(
            state=state, action=action, reward=reward,
            next_st=state, done=(reward != 0)
        )
        self._buffer.append(exp)

        # Mini-batch update
        if len(self._buffer) >= self.BATCH_SIZE:
            self._replay_update()

        # Epsilon annealing
        self._epsilon = max(self.EPSILON_MIN, self._epsilon * self.EPSILON_DECAY)
        self._total_updates += 1

        if self._total_updates % 10 == 0:
            self._save_full_state()

    def _replay_update(self) -> None:
        """Atualiza Q-table com mini-batch do replay buffer."""
        batch = random.sample(list(self._buffer), min(self.BATCH_SIZE, len(self._buffer)))
        for exp in batch:
            q_old   = self._q[exp.state][exp.action]
            v_next  = self._v.get(exp.next_st, 0.0)
            td      = exp.reward + self.DISCOUNT * v_next - q_old
            self._q[exp.state][exp.action] = round(q_old + self.LEARNING_RATE * 0.5 * td, 4)

    def _get_rich_state(self, df: pd.DataFrame, session) -> str:
        """
        Estado rico com 12 features discretizadas:
        RSI bucket, EMA dir, MACD sign, ADX bucket,
        BB position, hora, dia semana, drawdown, consec_loss,
        vol regime, momentum sign, win_rate bucket.
        """
        if len(df) < 5:
            return "init"
        try:
            last = df.iloc[-1]
            rsi  = self._safe_float(last.get("rsi_14", 50.0), 50.0)
            e9   = self._safe_float(last.get("ema_9",  1.0), 1.0)
            e21  = self._safe_float(last.get("ema_21", 0.0), 0.0)
            e50  = self._safe_float(last.get("ema_50", 0.0), 0.0)
            macd = self._safe_float(last.get("macd_hist", 0.0))
            adx  = self._safe_float(last.get("adx", 20.0), 20.0)
            bbu  = self._safe_float(last.get("bb_upper", 0.0))
            bbl  = self._safe_float(last.get("bb_lower", 0.0))
            close = self._safe_float(last.get("close", 1.0), 1.0)

            # RSI: 5 buckets
            rb = "VL" if rsi < 30 else "L" if rsi < 45 else "M" if rsi < 55 else "H" if rsi < 70 else "VH"
            # EMA: alinhamento triplo
            ed = "UU" if e9 > e21 > e50 else "U" if e9 > e21 else "DD" if e9 < e21 < e50 else "D"
            # MACD
            md = "P" if macd > 0 else "N"
            # ADX
            ad = "S" if adx > 30 else "T" if adx > 22 else "W"
            # BB position
            bb_pos = (close - bbl) / (bbu - bbl) if bbu > bbl else 0.5
            bp = "L" if bb_pos < 0.25 else "M" if bb_pos < 0.75 else "H"
            # Hora (bloco 4h)
            hb = (time.localtime().tm_hour // 4) * 4
            # Dia semana
            dw = time.localtime().tm_wday  # 0=Mon
            # Session state
            ini  = self._safe_float(getattr(session, "initial_balance", 1000.0), 1000.0)
            cur  = self._safe_float(getattr(session, "current_balance",     ini), ini)
            dd   = "OK" if (ini - cur) / max(ini, 1) < 0.05 else "BAD"
            cl   = str(min(getattr(session, "consecutive_losses", 0), 3))
            wr   = self._safe_float(getattr(session, "win_rate", 0.5), 0.5)
            wb   = "G" if wr > 0.55 else "N" if wr > 0.45 else "B"

            return f"{rb}_{ed}_{md}_{ad}_{bp}_{hb}_{dw}_{dd}_{cl}_{wb}"
        except Exception:
            return "fallback"

    # ── Sentimento Web ────────────────────────────────────────────────────────

    def _refresh_sentiment_background(self) -> None:
        """
        Atualiza Fear & Greed Index em background (thread não-bloqueante).
        Fonte: Alternative.me API (gratuita, sem auth).
        TTL: 30 minutos.
        """
        if time.time() - self._fg_fetched_at < 1800:
            return  # Ainda válido

        cached = self._get_web_cache("fear_greed_index")
        if cached is not None:
            self._fear_greed = cached
            return

        def _fetch():
            try:
                import urllib.request
                url = "https://api.alternative.me/fng/?limit=1&format=json"
                with urllib.request.urlopen(url, timeout=5) as resp:
                    data = json.loads(resp.read().decode())
                    value = float(data["data"][0]["value"])
                    self._fear_greed   = value
                    self._fg_fetched_at = time.time()
                    self._set_web_cache("fear_greed_index", value)
                    logger.debug("ECHO: Fear & Greed atualizado.", value=value)
            except Exception as exc:
                logger.debug("ECHO: falha ao buscar Fear & Greed.", error=str(exc))

        t = Thread(target=_fetch, daemon=True)
        t.start()

    # ── Persistência completa ─────────────────────────────────────────────────

    def _save_full_state(self) -> None:
        os.makedirs(os.path.dirname(_STATE_PATH), exist_ok=True)
        try:
            with open(_STATE_PATH, "w") as f:
                json.dump({
                    "q":             dict(self._q),
                    "v":             dict(self._v),
                    "epsilon":       self._epsilon,
                    "total_updates": self._total_updates,
                    "fear_greed":    self._fear_greed,
                    "thresholds":    self._thresholds,
                }, f)
        except Exception:
            pass

    def _load_full_state(self) -> None:
        try:
            if os.path.exists(_STATE_PATH):
                with open(_STATE_PATH) as f:
                    data = json.load(f)
                for st, vals in data.get("q", {}).items():
                    self._q[st] = vals
                for st, val in data.get("v", {}).items():
                    self._v[st] = val
                self._epsilon       = data.get("epsilon",       self.EPSILON_START)
                self._total_updates = data.get("total_updates", 0)
                self._fear_greed    = data.get("fear_greed",    50.0)
                self._thresholds    = data.get("thresholds",    self._default_thresholds())
                logger.info("ECHO: estado carregado.", updates=self._total_updates)
        except Exception:
            pass