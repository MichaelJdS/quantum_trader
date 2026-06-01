from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from backtesting.metrics import PerformanceMetrics
from backtesting.vectorized_backtester import BacktestResult


class ReportGenerator:
    """
    Gera relatório HTML interativo de resultados de backtesting.

    Inclui:
      - KPIs em cards coloridos.
      - Gráfico de equity curve interativo (Plotly via CDN).
      - Gráfico de drawdown.
      - Distribuição de PnL por trade (histograma).
      - Tabela comparativa de estratégias.
      - Tabela detalhada de todos os trades.
      - Radar chart de métricas por estratégia.
    """

    def __init__(self, output_dir: str = "./reports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        results: list[BacktestResult],
        title: str = "Quantum Trader — Backtest Report",
    ) -> Path:
        """
        Gera relatório HTML completo.

        Args:
            results: Lista de BacktestResult (uma ou mais estratégias).
            title: Título do relatório.

        Returns:
            Path do arquivo HTML gerado.
        """
        compare_df = PerformanceMetrics.compare_strategies(results)
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"backtest_report_{timestamp}.html"

        html = self._build_html(
            results=results,
            compare_df=compare_df,
            title=title,
            timestamp=timestamp,
        )
        path.write_text(html, encoding="utf-8")
        logger.success("Relatório gerado.", path=str(path))
        return path

    def _build_html(
        self,
        results: list[BacktestResult],
        compare_df: pd.DataFrame,
        title: str,
        timestamp: str,
    ) -> str:
        """Constrói o HTML completo do relatório."""

        # ── Dados para Plotly ──────────────────────────────────────────────
        equity_traces = self._build_equity_traces(results)
        drawdown_traces = self._build_drawdown_traces(results)
        pnl_dist_traces = self._build_pnl_dist_traces(results)
        radar_data = self._build_radar_data(results)
        compare_table_html = self._build_compare_table(compare_df)
        trades_tables_html = self._build_trades_tables(results)
        kpi_cards_html = self._build_kpi_cards(results)

        return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    {self._get_css()}
  </style>
</head>
<body data-theme="dark">

<header>
  <div class="header-inner">
    <div class="logo">
      <svg width="32" height="32" viewBox="0 0 32 32" fill="none" aria-label="Quantum Trader">
        <polygon points="16,2 30,28 2,28" stroke="#4f98a3" stroke-width="2" fill="none"/>
        <circle cx="16" cy="16" r="4" fill="#4f98a3"/>
        <line x1="16" y1="2" x2="16" y2="12" stroke="#4f98a3" stroke-width="1.5"/>
      </svg>
      <span>Quantum Trader</span>
    </div>
    <div class="header-meta">
      <span class="badge">Backtest Report</span>
      <span class="timestamp">{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[9:11]}:{timestamp[11:13]} UTC</span>
      <button class="theme-toggle" onclick="toggleTheme()" title="Alternar tema">🌙</button>
    </div>
  </div>
</header>

<main>
  <h1>{title}</h1>

  <!-- KPI Cards -->
  <section class="section">
    <h2>📊 Métricas Principais</h2>
    <div class="kpi-grid">
      {kpi_cards_html}
    </div>
  </section>

  <!-- Equity Curve -->
  <section class="section">
    <h2>📈 Equity Curve</h2>
    <div id="equity_chart" class="chart"></div>
  </section>

  <!-- Drawdown -->
  <section class="section">
    <h2>📉 Drawdown</h2>
    <div id="drawdown_chart" class="chart"></div>
  </section>

  <!-- PnL Distribution -->
  <section class="section">
    <h2>🎲 Distribuição de PnL por Trade</h2>
    <div id="pnl_dist_chart" class="chart-half"></div>
  </section>

  <!-- Radar -->
  <section class="section">
    <h2>🎯 Radar de Performance</h2>
    <div id="radar_chart" class="chart-half"></div>
  </section>

  <!-- Comparison Table -->
  <section class="section">
    <h2>⚔️ Comparativo de Estratégias</h2>
    {compare_table_html}
  </section>

  <!-- Trades Tables -->
  <section class="section">
    <h2>📋 Trades Detalhados</h2>
    {trades_tables_html}
  </section>
</main>

<footer>
  <p>Gerado por Quantum Trader APEX-CODER — {timestamp}</p>
  <p>⚠️ Resultados passados não garantem performance futura. Opere com responsabilidade.</p>
</footer>

<script>
{self._get_js(equity_traces, drawdown_traces, pnl_dist_traces, radar_data)}
</script>
</body>
</html>"""

    # ── Traces Plotly ──────────────────────────────────────────────────────

    def _build_equity_traces(self, results: list[BacktestResult]) -> str:
        traces = []
        colors = ["#4f98a3", "#6daa45", "#dd6974", "#fdab43", "#a86fdf"]
        for i, r in enumerate(results):
            color = colors[i % len(colors)]
            y = r.equity_curve.tolist()
            traces.append({
                "type": "scatter",
                "mode": "lines",
                "name": r.strategy_name,
                "y": y,
                "x": list(range(len(y))),
                "line": {"color": color, "width": 2},
                "hovertemplate": "$%{y:.2f}<extra>%{fullData.name}</extra>",
            })
        return json.dumps(traces)

    def _build_drawdown_traces(self, results: list[BacktestResult]) -> str:
        traces = []
        colors = ["#dd6974", "#fdab43", "#a86fdf", "#4f98a3", "#6daa45"]
        for i, r in enumerate(results):
            color = colors[i % len(colors)]
            y = (r.drawdown_series * 100).tolist()
            traces.append({
                "type": "scatter",
                "mode": "lines",
                "fill": "tozeroy",
                "name": r.strategy_name,
                "y": y,
                "x": list(range(len(y))),
                "line": {"color": color, "width": 1.5},
                "hovertemplate": "%{y:.2f}%<extra>%{fullData.name}</extra>",
            })
        return json.dumps(traces)

    def _build_pnl_dist_traces(self, results: list[BacktestResult]) -> str:
        traces = []
        colors = ["#4f98a3", "#6daa45", "#dd6974", "#fdab43"]
        for i, r in enumerate(results):
            color = colors[i % len(colors)]
            pnls = [t.pnl for t in r.trades]
            if not pnls:
                continue
            traces.append({
                "type": "histogram",
                "name": r.strategy_name,
                "x": pnls,
                "opacity": 0.75,
                "marker": {"color": color},
                "nbinsx": 30,
            })
        return json.dumps(traces)

    def _build_radar_data(self, results: list[BacktestResult]) -> str:
        RADAR_METRICS = [
            "win_rate", "profit_factor", "sharpe_ratio",
            "calmar_ratio", "expectancy_pct", "recovery_factor",
        ]
        RADAR_LABELS = [
            "Win Rate", "Profit Factor", "Sharpe",
            "Calmar", "Expectância%", "Recovery",
        ]

        def normalize(vals: list[float]) -> list[float]:
            mn, mx = min(vals), max(vals)
            if mx == mn:
                return [0.5] * len(vals)
            return [(v - mn) / (mx - mn + 1e-10) for v in vals]

        metric_vals: dict[str, list[float]] = {m: [] for m in RADAR_METRICS}
        for r in results:
            for m in RADAR_METRICS:
                metric_vals[m].append(r.metrics.get(m, 0.0))

        traces = []
        colors = ["#4f98a3", "#6daa45", "#dd6974", "#fdab43", "#a86fdf"]

        for i, r in enumerate(results):
            normalized = []
            for m in RADAR_METRICS:
                all_vals = metric_vals[m]
                n = normalize(all_vals)
                normalized.append(n[i])

            traces.append({
                "type": "scatterpolar",
                "r": normalized + [normalized[0]],
                "theta": RADAR_LABELS + [RADAR_LABELS[0]],
                "fill": "toself",
                "name": r.strategy_name,
                "line": {"color": colors[i % len(colors)]},
                "opacity": 0.7,
            })
        return json.dumps(traces)

    # ── Tabelas HTML ───────────────────────────────────────────────────────

    def _build_compare_table(self, df: pd.DataFrame) -> str:
        if df.empty:
            return "<p>Sem dados.</p>"

        DISPLAY_COLS = {
            "strategy": "Estratégia",
            "total_return_pct": "Retorno%",
            "sharpe_ratio": "Sharpe",
            "sortino_ratio": "Sortino",
            "calmar_ratio": "Calmar",
            "win_rate": "Win Rate",
            "profit_factor": "Profit Factor",
            "expectancy_pct": "Expectância%",
            "max_drawdown_pct": "Max DD%",
            "total_trades": "Trades",
        }

        cols = [c for c in DISPLAY_COLS.keys() if c in df.columns]
        headers = "".join(f"<th>{DISPLAY_COLS[c]}</th>" for c in cols)
        rows_html = ""

        for _, row in df[cols].iterrows():
            cells = ""
            for c in cols:
                val = row[c]
                css = ""
                if c in ("total_return_pct", "expectancy_pct", "sharpe_ratio"):
                    css = "positive" if float(val) >= 0 else "negative"
                if isinstance(val, float):
                    val = f"{val:.4f}"
                cells += f'<td class="{css}">{val}</td>'
            rows_html += f"<tr>{cells}</tr>"

        return f"""
<div class="table-wrap">
  <table class="data-table">
    <thead><tr>{headers}</tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>"""

    def _build_trades_tables(self, results: list[BacktestResult]) -> str:
        html = ""
        for r in results:
            trades = r.trades[:200]  # Máx 200 por tabela.
            if not trades:
                continue

            rows_html = ""
            for t in trades:
                pnl_css = "positive" if t.pnl >= 0 else "negative"
                icon = "🟢" if t.won else "🔴"
                dir_icon = "↑ CALL" if t.direction == "BUY" else "↓ PUT"
                rows_html += f"""<tr>
                  <td>{t.entry_idx}</td>
                  <td>{dir_icon}</td>
                  <td>${t.entry_price:.5f}</td>
                  <td>${t.exit_price:.5f}</td>
                  <td>${t.stake:.4f}</td>
                  <td class="{pnl_css}">{icon} ${t.pnl:+.4f}</td>
                  <td>{t.confidence:.1%}</td>
                </tr>"""

            html += f"""
<details open>
  <summary><strong>{r.strategy_name}</strong> — {len(r.trades)} trades</summary>
  <div class="table-wrap">
    <table class="data-table">
      <thead>
        <tr><th>Idx</th><th>Dir</th><th>Entrada</th><th>Saída</th>
            <th>Stake</th><th>PnL</th><th>Conf.</th></tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</details>"""
        return html

    def _build_kpi_cards(self, results: list[BacktestResult]) -> str:
        html = ""
        colors = ["#4f98a3", "#6daa45", "#dd6974", "#fdab43", "#a86fdf"]
        for i, r in enumerate(results):
            m = r.metrics
            color = colors[i % len(colors)]
            pnl_pct = m.get("total_return_pct", 0)
            pnl_sign = "+" if pnl_pct >= 0 else ""
            pnl_cls = "positive" if pnl_pct >= 0 else "negative"

            html += f"""
<div class="kpi-card" style="border-top: 3px solid {color}">
  <div class="kpi-strategy">{r.strategy_name}</div>
  <div class="kpi-row-inner">
    <div class="kpi-item">
      <div class="kpi-label">Retorno</div>
      <div class="kpi-value {pnl_cls}">{pnl_sign}{pnl_pct:.2f}%</div>
    </div>
    <div class="kpi-item">
      <div class="kpi-label">Sharpe</div>
      <div class="kpi-value">{m.get('sharpe_ratio', 0):.3f}</div>
    </div>
    <div class="kpi-item">
      <div class="kpi-label">Win Rate</div>
      <div class="kpi-value">{m.get('win_rate', 0):.1%}</div>
    </div>
    <div class="kpi-item">
      <div class="kpi-label">Max DD</div>
      <div class="kpi-value negative">-{m.get('max_drawdown_pct', 0):.2f}%</div>
    </div>
    <div class="kpi-item">
      <div class="kpi-label">Profit Factor</div>
      <div class="kpi-value">{m.get('profit_factor', 0):.3f}</div>
    </div>
    <div class="kpi-item">
      <div class="kpi-label">Trades</div>
      <div class="kpi-value">{int(m.get('total_trades', 0))}</div>
    </div>
  </div>
</div>"""
        return html

    # ── CSS / JS ───────────────────────────────────────────────────────────

    def _get_css(self) -> str:
        return """
    :root {
      --bg: #171614; --surface: #1c1b19; --surface-2: #201f1d;
      --border: #393836; --text: #cdccca; --text-muted: #797876;
      --primary: #4f98a3; --success: #6daa45; --error: #dd6974;
      --warning: #fdab43; --radius: 0.5rem;
    }
    [data-theme="light"] {
      --bg: #f7f6f2; --surface: #f9f8f5; --surface-2: #fbfbf9;
      --border: #d4d1ca; --text: #28251d; --text-muted: #7a7974;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: var(--bg); color: var(--text);
      font-size: 15px; line-height: 1.6;
    }
    header {
      background: var(--surface); border-bottom: 1px solid var(--border);
      position: sticky; top: 0; z-index: 100; padding: 0.75rem 2rem;
    }
    .header-inner {
      max-width: 1400px; margin: auto;
      display: flex; justify-content: space-between; align-items: center;
    }
    .logo { display: flex; align-items: center; gap: 0.75rem; font-weight: 700; font-size: 1.1rem; color: var(--primary); }
    .header-meta { display: flex; align-items: center; gap: 1rem; }
    .badge {
      background: var(--primary); color: #fff;
      font-size: 0.75rem; padding: 0.2rem 0.6rem;
      border-radius: 999px; font-weight: 600;
    }
    .timestamp { color: var(--text-muted); font-size: 0.85rem; }
    .theme-toggle {
      background: none; border: 1px solid var(--border);
      padding: 0.25rem 0.5rem; border-radius: var(--radius);
      cursor: pointer; color: var(--text); font-size: 1rem;
    }
    main { max-width: 1400px; margin: auto; padding: 2rem; }
    h1 { font-size: 1.8rem; color: var(--primary); margin-bottom: 2rem; }
    h2 { font-size: 1.15rem; color: var(--text); margin-bottom: 1rem; font-weight: 600; }
    .section { margin-bottom: 2.5rem; }
    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 1rem; }
    .kpi-card {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: var(--radius); padding: 1.25rem;
    }
    .kpi-strategy { font-weight: 700; margin-bottom: 0.75rem; color: var(--text-muted); font-size: 0.9rem; }
    .kpi-row-inner { display: flex; flex-wrap: wrap; gap: 1rem; }
    .kpi-item { flex: 1; min-width: 80px; }
    .kpi-label { font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.25rem; }
    .kpi-value { font-size: 1.1rem; font-weight: 700; font-variant-numeric: tabular-nums; }
    .positive { color: var(--success); }
    .negative { color: var(--error); }
    .chart { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 0.5rem; min-height: 350px; }
    .chart-half { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 0.5rem; min-height: 320px; }
    .table-wrap { overflow-x: auto; }
    .data-table {
      width: 100%; border-collapse: collapse; font-size: 0.85rem;
      background: var(--surface);
    }
    .data-table th {
      background: var(--surface-2); color: var(--text-muted);
      padding: 0.6rem 0.8rem; text-align: left;
      border-bottom: 1px solid var(--border); font-weight: 600;
      white-space: nowrap;
    }
    .data-table td {
      padding: 0.5rem 0.8rem; border-bottom: 1px solid var(--border);
      font-variant-numeric: tabular-nums;
    }
    .data-table tr:hover td { background: var(--surface-2); }
    details { margin-bottom: 1rem; }
    details summary {
      cursor: pointer; padding: 0.75rem 1rem;
      background: var(--surface); border: 1px solid var(--border);
      border-radius: var(--radius); margin-bottom: 0.5rem;
    }
    footer {
      text-align: center; padding: 2rem;
      color: var(--text-muted); font-size: 0.85rem;
      border-top: 1px solid var(--border); margin-top: 3rem;
    }
    @media (max-width: 768px) {
      main { padding: 1rem; }
      .kpi-grid { grid-template-columns: 1fr; }
    }
    """

    def _get_js(
        self,
        equity_traces: str,
        drawdown_traces: str,
        pnl_dist_traces: str,
        radar_data: str,
    ) -> str:
        return f"""
    const DARK_LAYOUT = {{
      paper_bgcolor: '#1c1b19', plot_bgcolor: '#1c1b19',
      font: {{ color: '#cdccca', family: 'Segoe UI, system-ui, sans-serif' }},
      xaxis: {{ gridcolor: '#393836', zerolinecolor: '#393836' }},
      yaxis: {{ gridcolor: '#393836', zerolinecolor: '#393836' }},
      legend: {{ bgcolor: '#201f1d', bordercolor: '#393836', borderwidth: 1 }},
      margin: {{ t: 40, r: 20, b: 40, l: 60 }},
    }};

    const LIGHT_LAYOUT = {{
      paper_bgcolor: '#f9f8f5', plot_bgcolor: '#f9f8f5',
      font: {{ color: '#28251d', family: 'Segoe UI, system-ui, sans-serif' }},
      xaxis: {{ gridcolor: '#d4d1ca', zerolinecolor: '#d4d1ca' }},
      yaxis: {{ gridcolor: '#d4d1ca', zerolinecolor: '#d4d1ca' }},
      legend: {{ bgcolor: '#fbfbf9', bordercolor: '#d4d1ca', borderwidth: 1 }},
      margin: {{ t: 40, r: 20, b: 40, l: 60 }},
    }};

    function getLayout() {{
      return document.body.dataset.theme === 'light' ? LIGHT_LAYOUT : DARK_LAYOUT;
    }}

    function plotAll() {{
      const layout = getLayout();

      Plotly.newPlot('equity_chart', {equity_traces}, {{
        ...layout, title: 'Equity Curve — Evolução do Saldo',
        yaxis: {{ ...layout.yaxis, title: 'Saldo ($)', tickprefix: '$' }},
        xaxis: {{ ...layout.xaxis, title: 'Candle' }},
      }}, {{ responsive: true }});

      Plotly.newPlot('drawdown_chart', {drawdown_traces}, {{
        ...layout, title: 'Drawdown (%)',
        yaxis: {{ ...layout.yaxis, title: 'Drawdown (%)', ticksuffix: '%' }},
        xaxis: {{ ...layout.xaxis, title: 'Candle' }},
      }}, {{ responsive: true }});

      Plotly.newPlot('pnl_dist_chart', {pnl_dist_traces}, {{
        ...layout, title: 'Distribuição de PnL por Trade',
        barmode: 'overlay',
        xaxis: {{ ...layout.xaxis, title: 'PnL ($)' }},
        yaxis: {{ ...layout.yaxis, title: 'Frequência' }},
      }}, {{ responsive: true }});

      Plotly.newPlot('radar_chart', {radar_data}, {{
        ...layout, title: 'Radar de Performance (Normalizado)',
        polar: {{ radialaxis: {{ visible: true, range: [0, 1] }} }},
      }}, {{ responsive: true }});
    }}

    function toggleTheme() {{
      const body = document.body;
      body.dataset.theme = body.dataset.theme === 'light' ? 'dark' : 'light';
      document.querySelector('.theme-toggle').textContent =
        body.dataset.theme === 'dark' ? '🌙' : '☀️';
      plotAll();
    }}

    document.addEventListener('DOMContentLoaded', plotAll);
    """