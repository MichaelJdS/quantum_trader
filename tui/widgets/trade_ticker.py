from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static

from tui.state import TUIState


class TradeTicker(Widget):
    """
    Ticker de últimos trades com PnL colorido.
    Exibe os 10 trades mais recentes em formato compacto.
    """

    DEFAULT_CSS = """
    TradeTicker { padding: 0 1; }
    """

    def __init__(self, state: TUIState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = state

    def compose(self) -> ComposeResult:
        yield Static(id="ticker_content")

    def refresh_data(self) -> None:
        trades = self.state.recent_trades[:10]
        if not trades:
            self.query_one("#ticker_content", Static).update(
                "Aguardando trades..."
            )
            return

        lines = []
        for t in trades:
            pnl = t.get("pnl", 0.0) or 0.0
            icon = "🟢" if pnl >= 0 else "🔴"
            dir_icon = "↑" if t.get("direction") == "BUY" else "↓"
            pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
            lines.append(
                f"{icon} {t.get('symbol',''):<6} {dir_icon} "
                f"${t.get('stake',0):.2f} → {pnl_str}"
            )

        self.query_one("#ticker_content", Static).update("\n".join(lines))