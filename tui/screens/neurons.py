from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import DataTable, Label, Rule, Static

from tui.state import TUIState


class NeuronsScreen(Widget):
    """
    Monitor de saúde das camadas do LSTM em tempo real.

    Exibe:
      - Tabela de estatísticas por camada (dead%, saturated%, mean, std).
      - Status de vanishing/exploding gradient.
      - Alertas ativos.
      - Métricas do OnlineLearner (accuracy, kappa, drift count).
      - Pesos do ensemble.
    """

    def __init__(self, state: TUIState) -> None:
        super().__init__()
        self.state = state
        self._initialized = False

    def compose(self) -> ComposeResult:
        yield Label("🧠 Monitor de Neurônios — LSTM", classes="section-title")
        yield DataTable(id="neurons_table", zebra_stripes=True)

        yield Rule()

        with Horizontal(id="gradient_row"):
            with Vertical(id="gradient_panel", classes="panel"):
                yield Label("📡 Gradientes", classes="section-title")
                yield Static(id="gradient_text")
            with Vertical(id="alerts_panel", classes="panel"):
                yield Label("⚠️ Alertas", classes="section-title")
                yield Static(id="neuron_alerts_text")

        yield Rule()

        with Horizontal(id="ml_row"):
            with Vertical(id="online_panel", classes="panel"):
                yield Label("📚 Online Learner", classes="section-title")
                yield Static(id="online_metrics_text")
            with Vertical(id="ensemble_panel", classes="panel"):
                yield Label("⚖️ Pesos do Ensemble", classes="section-title")
                yield Static(id="ensemble_weights_text")

    def on_mount(self) -> None:
        table = self.query_one("#neurons_table", DataTable)
        table.add_columns(
            "Camada", "Status", "Dead%", "Saturado%",
            "Média", "Desvio", "Neurônios", "Updates"
        )
        self._initialized = True
        self._refresh()
        self.set_interval(2.0, self._refresh)

    def _refresh(self) -> None:
        if not self._initialized:
            return

        summary = self.state.neuron_summary
        if not summary:
            return

        # Tabela de camadas.
        table = self.query_one("#neurons_table", DataTable)
        table.clear()
        layers = summary.get("layers", {})
        for name, stats in layers.items():
            status = stats.get("status", "?")
            status_icon = {
                "OK": "✅ OK",
                "ATENÇÃO": "⚠️ ATENÇÃO",
                "CRÍTICO": "❌ CRÍTICO",
                "SATURADO": "🟡 SAT",
            }.get(status, status)
            table.add_row(
                name,
                status_icon,
                f"{stats.get('dead_pct', 0):.1%}",
                f"{stats.get('saturated_pct', 0):.1%}",
                f"{stats.get('mean', 0):+.4f}",
                f"{stats.get('std', 0):.4f}",
                str(stats.get("neurons", 0)),
                str(stats.get("updates", 0)),
            )

        # Gradientes.
        vanishing = summary.get("vanishing_gradient", False)
        exploding = summary.get("exploding_gradient", False)
        grad_text = (
            f"Vanishing Gradient: {'❌ DETECTADO' if vanishing else '✅ OK'}\n"
            f"Exploding Gradient: {'❌ DETECTADO' if exploding else '✅ OK'}\n\n"
            "Gradientes por parâmetro:\n"
        )
        for param, val in list(summary.get("gradient_stats", {}).items())[:10]:
            icon = "⚠️" if val < 1e-7 or val > 10 else " "
            grad_text += f"{icon} {param[-30:]:<30}: {val:.2e}\n"
        self.query_one("#gradient_text", Static).update(grad_text)

        # Alertas.
        alerts = summary.get("alerts", [])
        alert_text = "\n".join(alerts) if alerts else "✅ Nenhum alerta ativo."
        self.query_one("#neuron_alerts_text", Static).update(alert_text)

        # Online Learner.
        online = self.state.online_metrics
        ol_text = ""
        for symbol, metrics in online.items():
            ol_text += (
                f"[{symbol}]\n"
                f"  Amostras : {metrics.get('total_learned', 0)}\n"
                f"  Accuracy : {metrics.get('accuracy', 0):.1%}\n"
                f"  F1 Score : {metrics.get('f1', 0):.4f}\n"
                f"  Kappa    : {metrics.get('kappa', 0):.4f}\n"
                f"  Drifts   : {metrics.get('total_drifts', 0)}\n\n"
            )
        self.query_one("#online_metrics_text", Static).update(
            ol_text or "Aguardando dados do learner..."
        )

        # Ensemble.
        weights = self.state.ensemble_weights
        w_text = ""
        for model, weight in sorted(weights.items(), key=lambda x: -x[1]):
            bar = "█" * int(weight * 30)
            w_text += f"{model:<14}: [{bar:<30}] {weight:.1%}\n"
        self.query_one("#ensemble_weights_text", Static).update(
            w_text or "Aguardando dados do ensemble..."
        )