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
    RiskGauge ProgressBar.gauge-warning Bar.-complete  { color: $warning; }
    RiskGauge ProgressBar.gauge-critical Bar.-complete { color: $error;   }
    """

    def __init__(self, label: str, max_value: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self._label = label
        self._max_value = max_value
        self._current_value = 0.0
        # FIX: IDs internos únicos por instância — evita TooManyMatches quando
        # múltiplos RiskGauge existem na mesma tela.
        # O widget_id vem do kwarg `id=` passado pelo pai (ex: "gauge_drawdown").
        # Se não houver id, usa id() do objeto como fallback único.
        _uid = kwargs.get("id") or str(id(self))
        self._bar_id = f"bar_{_uid}"
        self._label_id = f"label_{_uid}"

    def compose(self) -> ComposeResult:
        # FIX: Usa IDs únicos por instância ao invés de "#bar" e "#gauge_label"
        # fixos, que colidiiam entre instâncias do mesmo widget na árvore global.
        yield ProgressBar(
            id=self._bar_id,
            total=100,
            show_eta=False,
            classes="gauge-safe",
        )
        yield Label("", id=self._label_id)

    def update_value(self, ratio: float) -> None:
        """
        Atualiza o gauge com o ratio atual de utilização do limite.

        Args:
            ratio: 0.0 a 1.0  (0% = seguro, 1.0+ = limite atingido).
        """
        self._current_value = min(ratio, 1.0)
        pct = self._current_value * 100

        # FIX: query com ID único — sem risco de TooManyMatches.
        bar = self.query_one(f"#{self._bar_id}", ProgressBar)
        bar.progress = pct

        # Cor por nível de risco — remove todas antes de adicionar nova.
        bar.remove_class("gauge-safe", "gauge-warning", "gauge-critical")
        if pct < 60:
            bar.add_class("gauge-safe")
        elif pct < 85:
            bar.add_class("gauge-warning")
        else:
            bar.add_class("gauge-critical")

        label = self.query_one(f"#{self._label_id}", Label)
        label.update(f"{self._label}: {pct:.1f}%")