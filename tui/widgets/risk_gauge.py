from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Label, ProgressBar


class RiskGauge(Widget):
    """
    Gauge de utilização de risco com barra de progresso colorida.

    0-60%   → verde (seguro)
    60-85%  → amarelo (atenção)
    85-100% → vermelho (crítico)
    """

    DEFAULT_CSS = """
    RiskGauge {
        height: 4;
        margin: 0 0 1 0;
    }
    RiskGauge ProgressBar Bar.-complete { color: $success; }
    """

    def __init__(self, label: str, max_value: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self._label = label
        self._max_value = max_value
        self._current_value = 0.0

    def compose(self) -> ComposeResult:
        yield ProgressBar(id="bar", total=100, show_eta=False)
        yield Label(id="gauge_label")

    def update_value(self, ratio: float) -> None:
        """
        Args:
            ratio: 0.0 a 1.0 (0% = seguro, 1.0+ = limite atingido).
        """
        self._current_value = min(ratio, 1.0)
        pct = self._current_value * 100
        bar = self.query_one("#bar", ProgressBar)
        bar.progress = pct

        # Cor por nível de risco.
        bar.remove_class("gauge-safe", "gauge-warning", "gauge-critical")
        if pct < 60:
            bar.add_class("gauge-safe")
        elif pct < 85:
            bar.add_class("gauge-warning")
        else:
            bar.add_class("gauge-critical")

        label = self.query_one("#gauge_label", Label)
        label.update(f"{self._label}: {pct:.1f}%")