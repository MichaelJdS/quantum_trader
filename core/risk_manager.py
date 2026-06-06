from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

from loguru import logger

from core.entities import RiskConfig, SessionState, Trade
from core.enums import StakeMode
from core.exceptions import RiskViolationError, StopLossReachedError, StopWinReachedError


@dataclass
class RiskManager:
    """
    Gestão de risco profissional para o Quantum Trader.

    Responsabilidades:
      - Cálculo de stake via Fixed, Kelly, Fractional Kelly e Adaptive.
      - Bloqueio de operação por drawdown, perdas consecutivas.
      - Detecção de Stop Win e Stop Loss da sessão.
      - Registro imutável de trades para cálculo de métricas.
    """

    config: RiskConfig
    initial_balance: float

    # FIX: deque(maxlen=N) ao invés de list — pop(0) era O(n)
    _trades: deque[Trade] = field(default_factory=lambda: deque(maxlen=10_000), repr=False)

    # FIX B11: datetime.utcnow() deprecated desde Python 3.12 — usa aware datetime.
    _session_start: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc), repr=False
    )

    # FIX B8: high water mark real da sessão para cálculo correto de drawdown.
    _peak_balance: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._peak_balance = self.initial_balance

    # ── Stake Calculation ─────────────────────────────────────────────────────

    def calculate_stake(
        self,
        current_balance: float,
        win_probability: float | None = None,
        payout_multiplier: float = 2.0,
    ) -> float:
        """
        Calcula o stake ideal de acordo com o modo configurado.

        Args:
            current_balance:  Saldo atual da conta.
            win_probability:  Probabilidade de ganho (necessário para Kelly).
            payout_multiplier: Multiplicador de payout do contrato.

        Returns:
            Stake calculado, respeitando limites de exposição.
        """
        match self.config.stake_mode:
            case StakeMode.FIXED:
                stake = self.config.base_stake

            case StakeMode.KELLY:
                if win_probability is None:
                    logger.warning(
                        "Kelly solicitado sem probabilidade — fallback para FIXED."
                    )
                    stake = self.config.base_stake
                else:
                    result = self._kelly_stake(
                        current_balance, win_probability, payout_multiplier,
                        fraction=1.0,
                    )
                    # FIX: Kelly pode retornar None se payout inválido
                    stake = result if result is not None else self.config.base_stake

            case StakeMode.FRACTIONAL_KELLY:
                if win_probability is None:
                    logger.warning(
                        "Fractional Kelly solicitado sem probabilidade — fallback para FIXED."
                    )
                    stake = self.config.base_stake
                else:
                    result = self._kelly_stake(
                        current_balance, win_probability, payout_multiplier,
                        fraction=self.config.kelly_fraction,
                    )
                    # FIX: Kelly pode retornar None se payout inválido
                    stake = result if result is not None else self.config.base_stake

            case StakeMode.FRACTIONAL:
                stake = current_balance * self.config.kelly_fraction

            case StakeMode.ADAPTIVE:
                stake = self._adaptive_stake(current_balance)

            case _:
                logger.warning(
                    "StakeMode desconhecido — fallback para FIXED.",
                    mode=self.config.stake_mode,
                )
                stake = self.config.base_stake

        max_stake = current_balance * 0.02  # Nunca mais que 2% do saldo.
        final_stake = round(min(stake, max_stake), 2)

        # FIX B6: Alerta quando o mínimo da Deriv viola o limite de 2% da banca.
        min_deriv_stake = 0.35
        if final_stake < min_deriv_stake:
            if current_balance > 0 and (min_deriv_stake / current_balance) > 0.02:
                logger.warning(
                    "Stake mínimo da Deriv ($0.35) viola limite de 2% da banca. "
                    "Operação executada com exposição acima do limite configurado.",
                    balance=current_balance,
                    exposure_pct=round(min_deriv_stake / current_balance, 4),
                )
            final_stake = min_deriv_stake

        logger.debug(
            "Stake calculado.",
            mode=self.config.stake_mode.value,
            stake=final_stake,
            balance=current_balance,
        )
        return final_stake

    def _kelly_stake(
        self,
        balance: float,
        p: float,
        payout: float,
        fraction: float = 1.0,
    ) -> float | None:
        """
        Kelly Criterion com fração configurável: f* = (bp - q) / b * fraction.

        Retorna None se o payout for inválido (≤ 1.0), para que o caller
        possa fallback para stake base explicitamente.

        Args:
            balance:  Saldo atual.
            p:        Probabilidade de ganho (0.0 – 1.0).
            payout:   Multiplicador do payout do contrato (ex: 1.95).
            fraction: Fração de Kelly a aplicar (ex: 0.25 para Fractional Kelly).

        Returns:
            Stake calculado ou None se payout ≤ 1.0.
        """
        b = payout - 1.0

        # FIX: b ≤ 0 significa payout igual ou menor que o stake (inviável).
        # Antes, retornava 0.0 silenciosamente, gerando stake zerado sem aviso.
        if b <= 0:
            logger.warning(
                "Payout inválido para Kelly Criterion — não é possível calcular stake. "
                "Verifique o contrato selecionado ou use modo FIXED.",
                payout=payout,
                b=b,
            )
            return None

        q = 1.0 - p
        kelly_full = (b * p - q) / b
        kelly_frac = max(kelly_full, 0.0) * fraction
        return balance * kelly_frac

    def _adaptive_stake(self, balance: float) -> float:
        """
        Ajusta stake dinamicamente:
          - Reduz 50% após cada perda consecutiva.
          - Escala até 1.5x base após 3 ganhos seguidos.
        """
        consecutive_losses = self._count_consecutive_losses()
        if consecutive_losses > 0:
            factor = 0.5 ** consecutive_losses
        else:
            consecutive_wins = self._count_consecutive_wins()
            factor = min(1.0 + consecutive_wins * 0.1, 1.5)

        return self.config.base_stake * factor

    # ── Bloqueio de Operação ──────────────────────────────────────────────────

    def can_trade(self, session: SessionState) -> tuple[bool, str]:
        """
        Verifica todas as travas de risco antes de abrir operação.

        Returns:
            (True, "OK") se permitido, (False, motivo) caso contrário.
        """
        # 1. Stop Loss da sessão.
        if self._session_loss_pct(session) >= self.config.stop_loss_pct:
            return False, f"__STOP_LOSS__: Stop Loss da sessão atingido ({self.config.stop_loss_pct:.1%})"

        # 2. Stop Win da sessão.
        if self._session_gain_pct(session) >= self.config.stop_win_pct:
            return False, f"__STOP_WIN__: Stop Win da sessão atingido ({self.config.stop_win_pct:.1%})"

        # 3. Drawdown diário.
        dd = self._daily_drawdown(session)
        if dd >= self.config.max_daily_drawdown_pct:
            return False, f"__STOP_LOSS__: Max drawdown diário atingido ({dd:.1%})"

        # 4. Perdas consecutivas.
        consec = self._count_consecutive_losses()
        if consec >= self.config.max_consecutive_losses:
            return False, f"__STOP_LOSS__: {consec} perdas consecutivas — pausa obrigatória"

        return True, "OK"

    def assert_can_trade(self, session: SessionState) -> None:
        """Levanta exceção tipada se operação for bloqueada."""
        ok, reason = self.can_trade(session)
        if not ok:
            # FIX B16: Usa prefixos estruturados ao invés de string matching frágil.
            if reason.startswith("__STOP_WIN__"):
                raise StopWinReachedError(reason)
            if reason.startswith("__STOP_LOSS__"):
                raise StopLossReachedError(reason)
            raise RiskViolationError(reason)

    # ── Registro de Trades ────────────────────────────────────────────────────

    def register_trade(self, trade: Trade) -> None:
        """Adiciona trade ao histórico interno."""
        self._trades.append(trade)

    def update_peak(self, current_balance: float) -> None:
        """Atualiza high water mark do saldo. Deve ser chamado após cada trade."""
        if current_balance > self._peak_balance:
            self._peak_balance = current_balance

    # ── Métricas ──────────────────────────────────────────────────────────────

    def session_metrics(self, session: SessionState) -> dict[str, float]:
        """
        Retorna métricas da sessão em formato serializável.

        Notas metodológicas:
          - sharpe_per_trade: Ratio de Sharpe calculado por trade (não anualizado
            por 252 dias), pois o bot opera intraday em granularidades de ticks/minutos.
            Anualizar por sqrt(252) seria estatisticamente incorreto aqui.
          - profit_factor: soma dos ganhos / soma das perdas (absoluto).
            Retorna inf se não houver perdas (proteção vs ZeroDivisionError).
        """
        trades_list = list(self._trades)
        pnls = [t.pnl for t in trades_list]
        wins = [p for p in pnls if p > 0]
        losses_list = [p for p in pnls if p <= 0]

        win_rate = len(wins) / len(pnls) if pnls else 0.0

        profit_factor = (
            sum(wins) / abs(sum(losses_list))
            if losses_list and sum(losses_list) != 0
            else float("inf")
        )

        avg_pnl = sum(pnls) / len(pnls) if pnls else 0.0

        # FIX: Desvio-padrão amostral (n-1) ao invés de populacional (n),
        # pois estamos estimando a partir de uma amostra de trades — não da
        # distribuição completa.
        pnl_std = (
            math.sqrt(
                sum((p - avg_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
            )
            if len(pnls) > 1
            else 0.0
        )

        # FIX: Sharpe por trade — não anualizado.
        # Usar sqrt(252) aqui seria incorreto pois as "observações" são trades,
        # não retornos diários. Se quiser anualização futura, passe o número de
        # trades por dia como parâmetro e use sqrt(trades_por_ano).
        sharpe_per_trade = (avg_pnl / pnl_std) if pnl_std > 0 else 0.0

        return {
            "total_trades":        len(pnls),
            "win_rate":            round(win_rate, 4),
            "profit_factor":       round(profit_factor, 4),
            "pnl_total":           round(sum(pnls), 4),
            "pnl_avg":             round(avg_pnl, 4),
            "sharpe_per_trade":    round(sharpe_per_trade, 4),
            "max_drawdown_pct":    round(self._daily_drawdown(session), 4),
            "consecutive_losses":  self._count_consecutive_losses(),
        }

    # ── Helpers Privados ──────────────────────────────────────────────────────

    def _session_loss_pct(self, session: SessionState) -> float:
        delta = session.initial_balance - session.current_balance
        return delta / session.initial_balance if session.initial_balance > 0 else 0.0

    def _session_gain_pct(self, session: SessionState) -> float:
        delta = session.current_balance - session.initial_balance
        return delta / session.initial_balance if session.initial_balance > 0 else 0.0

    def _daily_drawdown(self, session: SessionState) -> float:
        # FIX B8: Usa _peak_balance real (high water mark) ao invés de initial_balance.
        # O peak é atualizado via update_peak() após cada trade pelo ExecutionEngine.
        peak = max(self._peak_balance, session.current_balance)
        return (peak - session.current_balance) / peak if peak > 0 else 0.0

    def _count_consecutive_losses(self) -> int:
        count = 0
        for trade in reversed(self._trades):
            if trade.pnl < 0:
                count += 1
            else:
                break
        return count

    def _count_consecutive_wins(self) -> int:
        count = 0
        for trade in reversed(self._trades):
            if trade.pnl > 0:
                count += 1
            else:
                break
        return count