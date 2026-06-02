from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widget import Widget
from textual.widgets import Label, Rule, Static

from core.entities import RiskConfig
from tui.state import TUIState
from tui.widgets.risk_gauge import RiskGauge


class RiskScreen(Widget):
    """
    Monitoramento de risco em tempo real.

    Painéis:
      - Configuração atual (stop loss, stop win, drawdown máx).
      - Medidores visuais de utilização de risco.
      - Alerta de sequência de perdas.
    """

    DEFAULT_CSS = """
    RiskScreen {
        height: 1fr;
        overflow-y: auto;
        padding: 1 2;
    }
    RiskScreen .section-title {
        text-style: bold;
        color: $primary;
        margin-top: 1;
    }
    RiskScreen .config-grid {
        margin: 1 0;
        padding: 1;
        background: $surface;
        border: solid $primary;
    }
    """

    def __init__(self, state: TUIState, risk_config: RiskConfig) -> None:
        super().__init__()
        self.state = state
        self.config = risk_config

    def compose(self) -> ComposeResult:
        yield Label("🛡️ Gestão de Risco — Configuração Atual", classes="section-title")

        with Vertical(classes="config-grid"):
            yield Static(id="config_text")

        yield Rule()
        yield Label("📊 Utilização do Risco", classes="section-title")

        with Vertical(id="gauges"):
            # FIX: Labels duplicados removidos — RiskGauge já renderiza
            # seu próprio label interno com o nome + percentual.
            yield RiskGauge(id="gauge_drawdown", label="Drawdown Sessão", max_value=1.0)
            yield RiskGauge(id="gauge_stoploss", label="Stop Loss",        max_value=1.0)
            yield RiskGauge(id="gauge_stopwin",  label="Stop Win",         max_value=1.0)

        yield Rule()
        yield Label("⚠️ Alertas Ativos", classes="section-title")
        yield Static(id="risk_alerts")

    def on_mount(self) -> None:
        self._render_config()
        # FIX: call_after_refresh garante que o DOM esteja completamente
        # montado antes de refresh_data tentar fazer queries nos widgets filhos.
        # Chamar diretamente em on_mount pode falhar se os filhos ainda não
        # foram compostos (race condition na inicialização da árvore).
        self.call_after_refresh(self.refresh_data)

    def _render_config(self) -> None:
        text = (
            f"Stake Mode       : {self.config.stake_mode.value}\n"
            f"Base Stake       : ${self.config.base_stake:.2f}\n"
            f"Stop Loss        : {self.config.stop_loss_pct:.1%} da banca\n"
            f"Stop Win         : {self.config.stop_win_pct:.1%} da banca\n"
            f"Max Drawdown     : {self.config.max_daily_drawdown_pct:.1%} diário\n"
            f"Max Cons. Perdas : {self.config.max_consecutive_losses}\n"
            f"Kelly Fração     : {getattr(self.config, 'kelly_fraction', 0.25):.0%}\n"
        )
        self.query_one("#config_text", Static).update(text)

    def refresh_data(self) -> None:
        initial = self.state.initial_balance or 1.0
        current = self.state.balance

        drawdown_pct = max(0.0, (initial - current) / initial)
        stopwin_pct  = max(0.0, (current - initial) / initial)

        self.query_one("#gauge_drawdown", RiskGauge).update_value(
            drawdown_pct / (self.config.max_daily_drawdown_pct or 0.05)
        )
        self.query_one("#gauge_stoploss", RiskGauge).update_value(
            drawdown_pct / (self.config.stop_loss_pct or 0.03)
        )
        self.query_one("#gauge_stopwin", RiskGauge).update_value(
            stopwin_pct / (self.config.stop_win_pct or 0.05)
        )

        # Alertas.
        alerts: list[str] = []
        if self.state.consecutive_losses >= self.config.max_consecutive_losses:
            alerts.append(
                f"⛔ LIMITE DE PERDAS CONSECUTIVAS ATINGIDO: "
                f"{self.state.consecutive_losses}"
            )
        if drawdown_pct >= self.config.stop_loss_pct * 0.80:
            alerts.append(
                f"⚠️ Stop Loss próximo: drawdown atual {drawdown_pct:.2%} "
                f"(limite: {self.config.stop_loss_pct:.2%})"
            )
        if stopwin_pct >= self.config.stop_win_pct * 0.90:
            alerts.append(f"✅ Stop Win quase atingido: +{stopwin_pct:.2%}")
        if not alerts:
            alerts.append("✅ Todos os limites de risco dentro do normal.")

        self.query_one("#risk_alerts", Static).update("\n".join(alerts))