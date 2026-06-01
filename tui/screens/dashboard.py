from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Grid, Horizontal, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Label, Rule, Sparkline, Static

from tui.state import TUIState
from tui.widgets.kpi_card import KPICard
from tui.widgets.pnl_sparkline import PnLSparkline
from tui.widgets.trade_ticker import TradeTicker


class DashboardScreen(Widget):
    """
    Dashboard principal â€” atualizado a cada segundo.

    Layout:
    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
    â”‚ Saldo    â”‚ PnL      â”‚ Win Rate â”‚ Trades   â”‚
    â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
    â”‚            Sparkline PnL                   â”‚
    â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
    â”‚   MÃ©tricas de Risco  â”‚   Ãšltimos Trades    â”‚
    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
    """

    DEFAULT_CSS = """
    DashboardScreen {
        layout: vertical;
        padding: 1 2;
    }
    """

    def __init__(self, state: TUIState) -> None:
        super().__init__()
        self.state = state

    def compose(self) -> ComposeResult:
        # â”€â”€ Linha de KPIs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        with Horizontal(id="kpi_row", classes="kpi-row"):
            yield KPICard(
                id="kpi_balance",
                title="ðŸ’° Saldo",
                value=self.state.balance_str,
                subtitle="Conta atual",
            )
            yield KPICard(
                id="kpi_pnl",
                title="ðŸ“ˆ PnL SessÃ£o",
                value=self.state.pnl_str,
                subtitle="Desde o inÃ­cio",
                color_value=True,
            )
            yield KPICard(
                id="kpi_winrate",
                title="ðŸŽ¯ Win Rate",
                value=self.state.win_rate_str,
                subtitle=f"{self.state.wins}W / {self.state.losses}L",
            )
            yield KPICard(
                id="kpi_trades",
                title="ðŸ”¢ Trades",
                value=str(self.state.total_trades),
                subtitle=f"SequÃªncia perdas: {self.state.consecutive_losses}",
            )
            yield KPICard(
                id="kpi_drawdown",
                title="ðŸ“‰ Max Drawdown",
                value=f"{self.state.max_drawdown:.2%}",
                subtitle="MÃ¡ximo histÃ³rico sessÃ£o",
            )
            yield KPICard(
                id="kpi_duration",
                title="â±ï¸ DuraÃ§Ã£o",
                value=self.state.session_duration,
                subtitle="HH:MM:SS",
            )

        yield Rule()

        # â”€â”€ Sparkline PnL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        with Container(id="sparkline_container"):
            yield Label("ðŸ“Š EvoluÃ§Ã£o do PnL", classes="section-title")
            yield PnLSparkline(id="pnl_sparkline", state=self.state)

        yield Rule()

        # â”€â”€ MÃ©tricas + Ticker â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        with Horizontal(id="bottom_row"):
            with Vertical(id="risk_metrics_panel", classes="panel"):
                yield Label("ðŸ›¡ï¸ MÃ©tricas de Risco", classes="section-title")
                yield Static(id="risk_metrics_text")
            with Vertical(id="trade_ticker_panel", classes="panel"):
                yield Label("âš¡ Ãšltimos Trades", classes="section-title")
                yield TradeTicker(id="trade_ticker", state=self.state)

    def refresh_data(self) -> None:
        """Atualiza todos os widgets com dados novos do estado."""
        # KPIs.
        self.query_one("#kpi_balance", KPICard).update_value(self.state.balance_str)
        self.query_one("#kpi_pnl", KPICard).update_value(self.state.pnl_str)
        self.query_one("#kpi_winrate", KPICard).update_value(self.state.win_rate_str)
        self.query_one("#kpi_winrate", KPICard).update_subtitle(
            f"{self.state.wins}W / {self.state.losses}L"
        )
        self.query_one("#kpi_trades", KPICard).update_value(str(self.state.total_trades))
        self.query_one("#kpi_drawdown", KPICard).update_value(
            f"{self.state.max_drawdown:.2%}"
        )
        self.query_one("#kpi_duration", KPICard).update_value(self.state.session_duration)

        # Sparkline.
        self.query_one(PnLSparkline).refresh_data()

        # MÃ©tricas de risco.
        risk = self.state.risk_metrics
        text = (
            f"Sharpe Ratio   : {self.state.sharpe_ratio:+.3f}\n"
            f"Max Drawdown   : {self.state.max_drawdown:.2%}\n"
            f"PnL/Risco      : {risk.get('pnl_over_risk', 0.0):.3f}\n"
            f"Stop Loss      : {risk.get('stop_loss_threshold', 0.0):.2%}\n"
            f"Stop Win       : {risk.get('stop_win_threshold', 0.0):.2%}\n"
            f"Consecutivas   : {self.state.consecutive_losses} perdas\n"
            f"Modo           : {'DRY-RUN' if self.state.dry_run else 'ðŸ”´ LIVE'}\n"
        )
        self.query_one("#risk_metrics_text", Static).update(text)

        # Ticker.
        self.query_one(TradeTicker).refresh_data()
