from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, TabbedContent, TabPane

from tui.screens.dashboard import DashboardScreen
from tui.screens.history import HistoryScreen
from tui.screens.neurons import NeuronsScreen
from tui.screens.risk import RiskScreen
from tui.screens.symbols import SymbolsScreen
from tui.screens.trades import TradesScreen
from tui.state import TUIState

if TYPE_CHECKING:
    from core.execution_engine import ExecutionEngine


class QuantumTraderApp(App):
    """
    TUI principal do Quantum Trader.

    Abas:
      F1 — Dashboard  : visão geral em tempo real (PnL, win rate, saldo).
      F2 — Símbolos   : candles, indicadores e sinal por símbolo.
      F3 — Trades     : trades abertos e recentes.
      F4 — Histórico  : tabela completa de trades com filtros.
      F5 — Risco      : configuração e monitoramento de risco em tempo real.
      F6 — Neurônios  : ativações e health das camadas do LSTM.

    Atualização:
      Tick de UI a cada 1 segundo via set_interval.
      Dados vindos do TUIState compartilhado com o ExecutionEngine.
    """

    CSS_PATH = "styles/main.tcss"
    TITLE = "⚡ Quantum Trader"
    SUB_TITLE = "Deriv Automated Trading System"

    BINDINGS = [
        Binding("f1", "show_tab('dashboard')", "Dashboard", show=True),
        Binding("f2", "show_tab('symbols')", "Símbolos", show=True),
        Binding("f3", "show_tab('trades')", "Trades", show=True),
        Binding("f4", "show_tab('history')", "Histórico", show=True),
        Binding("f5", "show_tab('risk')", "Risco", show=True),
        Binding("f6", "show_tab('neurons')", "Neurônios", show=True),
        Binding("ctrl+p", "toggle_pause", "Pausar/Retomar", show=True),
        Binding("ctrl+e", "emergency_stop", "Stop de Emergência", show=True),
        Binding("ctrl+q", "quit", "Sair", show=True),
        Binding("d", "toggle_dark", "Tema", show=False),
    ]

    def __init__(self, engine: "ExecutionEngine", state: TUIState) -> None:
        super().__init__()
        self.engine = engine
        self.state = state
        self._paused = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(initial="dashboard", id="main_tabs"):
            with TabPane("📊 Dashboard", id="dashboard"):
                yield DashboardScreen(self.state)
            with TabPane("📈 Símbolos", id="symbols"):
                yield SymbolsScreen(self.state)
            with TabPane("💼 Trades", id="trades"):
                yield TradesScreen(self.state)
            with TabPane("📋 Histórico", id="history"):
                yield HistoryScreen(self.state)
            with TabPane("🛡️ Risco", id="risk"):
                yield RiskScreen(self.state, self.engine.risk_config)
            with TabPane("🧠 Neurônios", id="neurons"):
                yield NeuronsScreen(self.state)
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(1.0, self._refresh_all)
        self.set_interval(0.5, self._refresh_header)

    async def _refresh_all(self) -> None:
        """Atualiza todas as telas com o estado mais recente."""
        if self._paused:
            return
        self.state.sync_from_engine(self.engine)
        self.query_one(DashboardScreen).refresh_data()
        self.query_one(TradesScreen).refresh_data()
        self.query_one(RiskScreen).refresh_data()

    async def _refresh_header(self) -> None:
        """Atualiza subtítulo com status de conexão e modo."""
        mode = "DRY-RUN" if self.engine.dry_run else "🔴 LIVE"
        status = "▶ RODANDO" if self.engine.is_running else "⏸ PAUSADO"
        self.sub_title = f"{mode} | {status} | Saldo: {self.state.balance_str}"

    def action_show_tab(self, tab_id: str) -> None:
        self.query_one(TabbedContent).active = tab_id

    async def action_toggle_pause(self) -> None:
        self._paused = not self._paused
        status = "PAUSADO" if self._paused else "RETOMADO"
        self.notify(f"Sistema {status}.", severity="warning" if self._paused else "information")

    async def action_emergency_stop(self) -> None:
        """Para o engine imediatamente e exibe confirmação."""
        from textual.widgets import Button
        from tui.screens.confirm_stop import ConfirmStopScreen
        await self.push_screen(ConfirmStopScreen(), self._on_stop_confirmed)

    async def _on_stop_confirmed(self, confirmed: bool) -> None:
        if confirmed:
            await self.engine.stop()
            self.notify(
                "⛔ Sistema parado por emergência.",
                severity="error",
                timeout=10,
            )

    def action_toggle_dark(self) -> None:
        self.dark = not self.dark