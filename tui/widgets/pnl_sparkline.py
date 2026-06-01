from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Sparkline

from tui.state import TUIState


class PnLSparkline(Widget):
    """Sparkline do PnL acumulado ao longo dos trades da sessão."""

    def __init__(self, state: TUIState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = state
        self._data: list[float] = []

    def compose(self) -> ComposeResult:
        yield Sparkline(data=[0.0], id="sparkline_inner", summary_function=max)

    def refresh_data(self) -> None:
        trades = self.state.recent_trades
        if not trades:
            return

        # Acumula PnL progressivamente.
        cumulative = []
        total = 0.0
        for t in trades:
            total += t.get("pnl", 0.0) or 0.0
            cumulative.append(total)

        self._data = cumulative
        self.query_one("#sparkline_inner", Sparkline).data = cumulative