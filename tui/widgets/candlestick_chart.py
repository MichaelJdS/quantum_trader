from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static

from tui.state import TUIState


class CandlestickChart(Widget):
    """
    Gráfico de candles ASCII art no terminal.

    Representa os últimos 40 candles com:
      '█' verde para candle de alta.
      '█' vermelho para candle de baixa.
      '|' para o range total (high/low).
    """

    HEIGHT = 15
    WIDTH = 60

    def __init__(self, state: TUIState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = state
        self._candles: list[dict] = []
        self._symbol: str = ""

    def compose(self) -> ComposeResult:
        yield Static(id="chart_ascii")

    def update_symbol(self, symbol: str, candles: list[dict]) -> None:
        self._symbol = symbol
        self._candles = candles[-self.WIDTH :]
        self._render()

    def _render(self) -> None:
        candles = self._candles
        if not candles:
            self.query_one("#chart_ascii", Static).update(
                "Sem dados de candles para exibir."
            )
            return

        highs = [float(c.get("high", c.get("close", 0))) for c in candles]
        lows = [float(c.get("low", c.get("close", 0))) for c in candles]
        closes = [float(c.get("close", 0)) for c in candles]
        opens = [float(c.get("open", closes[i])) for i, c in enumerate(candles)]

        price_max = max(highs) if highs else 1
        price_min = min(lows) if lows else 0
        price_range = price_max - price_min or 1

        rows: list[list[str]] = [[" "] * len(candles) for _ in range(self.HEIGHT)]

        for col, (o, h, l, c) in enumerate(zip(opens, highs, lows, closes)):
            high_row = int((price_max - h) / price_range * (self.HEIGHT - 1))
            low_row = int((price_max - l) / price_range * (self.HEIGHT - 1))
            open_row = int((price_max - o) / price_range * (self.HEIGHT - 1))
            close_row = int((price_max - c) / price_range * (self.HEIGHT - 1))

            body_top = min(open_row, close_row)
            body_bot = max(open_row, close_row)
            is_bull = c >= o

            # Wick.
            for row in range(high_row, low_row + 1):
                if 0 <= row < self.HEIGHT:
                    rows[row][col] = "│"

            # Body.
            for row in range(body_top, body_bot + 1):
                if 0 <= row < self.HEIGHT:
                    rows[row][col] = "▲" if is_bull else "▼"

        lines = ["".join(row) for row in rows]

        # Escala de preço à esquerda.
        scale_lines = []
        for i, line in enumerate(lines):
            price = price_max - (i / (self.HEIGHT - 1)) * price_range
            scale_lines.append(f"{price:>8.2f} │{line}")

        chart_text = f"  {self._symbol} — últimos {len(candles)} candles\n"
        chart_text += "\n".join(scale_lines)
        self.query_one("#chart_ascii", Static).update(chart_text)