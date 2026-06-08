"""
desktop_app/screens/dashboard.py — Dashboard principal redesenhado
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QScrollArea, QSizePolicy, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget, QTextEdit, QGridLayout
)

from desktop_app.widgets.donut_chart import DonutChart
from desktop_app.widgets.equity_chart import EquityChart
from desktop_app.widgets.agent_card import AgentCard


class MetricCard(QWidget):
    def __init__(self, label: str, value: str = "—", subtitle: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("metric_card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(2)

        self._label = QLabel(label.upper())
        self._label.setObjectName("metric_label")
        layout.addWidget(self._label)

        self._value_label = QLabel(value)
        self._value_label.setObjectName("metric_value")
        font = QFont("Inter", 24, QFont.Weight.Bold)
        self._value_label.setFont(font)
        layout.addWidget(self._value_label)

        self._sub_label = QLabel(subtitle)
        self._sub_label.setStyleSheet("color: #f85149; font-size: 11px;")
        layout.addWidget(self._sub_label)

    def update_value(self, value: str, sub_val: str = "", color: str = "#e6edf3", sub_color: str = "#8b949e"):
        self._value_label.setText(value)
        self._value_label.setStyleSheet(f"color: {color};")
        if sub_val:
            self._sub_label.setText(sub_val)
            self._sub_label.setStyleSheet(f"color: {sub_color}; font-size: 11px;")


class DashboardScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Scroll area for entire dashboard
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        # ── Header ──
        header = QHBoxLayout()
        title = QLabel("Monitor em Tempo Real")
        title.setObjectName("section_title")
        title.setStyleSheet("font-size: 20px; font-weight: 700; color: #e6edf3;")
        
        badge = QLabel("● AO VIVO")
        badge.setStyleSheet("background-color: #1a4731; color: #3fb950; border-radius: 4px; padding: 4px 8px; font-size: 11px; font-weight: 700;")
        
        header.addWidget(title)
        header.addWidget(badge)
        header.addStretch()
        
        self.clock_label = QLabel("--:--:--")
        self.clock_label.setStyleSheet("color: #8b949e; font-size: 13px; font-family: 'Cascadia Code', monospace;")
        header.addWidget(self.clock_label)
        
        layout.addLayout(header)

        # ── Row 1: Metrics ──
        cards_row = QHBoxLayout()
        cards_row.setSpacing(12)
        
        self._card_balance = MetricCard("Saldo Total", "$0.00", "-0.00%")
        self._card_pnl = MetricCard("PNL do Dia", "$0.00", "0 trades")
        self._card_winrate = MetricCard("Win Rate", "0.0%", "0W / 0L")
        self._card_dd = MetricCard("Drawdown", "0.00%", "máx. dia")
        self._card_tps = MetricCard("Throughput", "0", "ticks/s")
        
        for card in [self._card_balance, self._card_pnl, self._card_winrate, self._card_dd, self._card_tps]:
            cards_row.addWidget(card)
        layout.addLayout(cards_row)

        # ── Row 2: Charts ──
        charts_row = QHBoxLayout()
        charts_row.setSpacing(20)
        
        # Equity Curve
        equity_container = QWidget()
        equity_container.setObjectName("metric_card")
        eq_layout = QVBoxLayout(equity_container)
        eq_header = QHBoxLayout()
        eq_title = QLabel("📈 Curva de Equity")
        eq_title.setStyleSheet("color: #e6edf3; font-weight: 700; font-size: 13px;")
        eq_header.addWidget(eq_title)
        eq_header.addStretch()
        eq_layout.addLayout(eq_header)
        
        self.equity_chart = EquityChart()
        eq_layout.addWidget(self.equity_chart)
        charts_row.addWidget(equity_container, stretch=7)
        
        # Grand Oracle
        oracle_container = QWidget()
        oracle_container.setObjectName("metric_card")
        or_layout = QVBoxLayout(oracle_container)
        
        or_title = QLabel("⚡ Grand Oracle")
        or_title.setStyleSheet("color: #e6edf3; font-weight: 700; font-size: 13px;")
        or_layout.addWidget(or_title)
        
        self.donut_chart = DonutChart()
        or_layout.addWidget(self.donut_chart, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.oracle_verdict = QLabel("Veredicto Final\nNEUTRAL")
        self.oracle_verdict.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.oracle_verdict.setStyleSheet("color: #8b949e; font-weight: 600; font-size: 12px;")
        or_layout.addWidget(self.oracle_verdict)
        
        # Stake & Sigma
        ss_layout = QGridLayout()
        ss_layout.addWidget(QLabel("Stake Kelly (25%)"), 0, 0)
        self.stake_label = QLabel("$0.00")
        self.stake_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        ss_layout.addWidget(self.stake_label, 0, 1)
        
        ss_layout.addWidget(QLabel("SIGMA Status"), 1, 0)
        self.sigma_status = QLabel("PERMITIDO")
        self.sigma_status.setStyleSheet("color: #3fb950; font-weight: 700;")
        self.sigma_status.setAlignment(Qt.AlignmentFlag.AlignRight)
        ss_layout.addWidget(self.sigma_status, 1, 1)
        
        or_layout.addLayout(ss_layout)
        charts_row.addWidget(oracle_container, stretch=3)
        
        layout.addLayout(charts_row)

        # ── Row 3: Conselheiros (Horizontal Scroll) ──
        agents_container = QWidget()
        agents_container.setObjectName("metric_card")
        ag_layout = QVBoxLayout(agents_container)
        
        ag_title = QLabel("👥 Conselheiros Ativos")
        ag_title.setStyleSheet("color: #e6edf3; font-weight: 700; font-size: 13px;")
        ag_layout.addWidget(ag_title)
        
        # Horizontal Scroll Area
        self.agents_scroll = QScrollArea()
        self.agents_scroll.setWidgetResizable(True)
        self.agents_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.agents_scroll.setFixedHeight(120)
        self.agents_scroll.setStyleSheet("QScrollArea { background: transparent; } QScrollBar:horizontal { height: 8px; }")
        
        self.agents_content = QWidget()
        self.agents_row = QHBoxLayout(self.agents_content)
        self.agents_row.setContentsMargins(0, 0, 0, 0)
        self.agents_row.setSpacing(12)
        
        # Will dynamically populate agent cards here
        self.agent_cards = {}
        
        # Pre-populate empty agent cards
        for ag in ["SIGMA", "SERAPH", "KRONOS", "VECTOR", "NEXUS", "ARES", "ECHO", "LUMEN", "GEMINI"]:
            card = AgentCard(ag)
            self.agents_row.addWidget(card)
            self.agent_cards[ag] = card
            
        self.agents_row.addStretch()
        
        self.agents_scroll.setWidget(self.agents_content)
        ag_layout.addWidget(self.agents_scroll)
        
        layout.addWidget(agents_container)

        # ── Row 4: Risco, Trades e Logs ──
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(20)
        
        # SIGMA Risk
        risk_container = QWidget()
        risk_container.setObjectName("metric_card")
        risk_layout = QVBoxLayout(risk_container)
        risk_title = QLabel("🛡️ SIGMA — Controle de Risco")
        risk_title.setStyleSheet("color: #e6edf3; font-weight: 700; font-size: 13px;")
        risk_layout.addWidget(risk_title)
        
        grid = QGridLayout()
        grid.addWidget(QLabel("PnL Diário"), 0, 0)
        self.sigma_pnl = QLabel("+$0.00")
        self.sigma_pnl.setStyleSheet("color: #3fb950; font-weight: 700;")
        grid.addWidget(self.sigma_pnl, 1, 0)
        
        grid.addWidget(QLabel("Sequência Perdas"), 0, 1)
        self.sigma_loss = QLabel("0 / 5")
        self.sigma_loss.setStyleSheet("font-weight: 700;")
        grid.addWidget(self.sigma_loss, 1, 1)
        
        grid.addWidget(QLabel("Trades Hoje"), 2, 0)
        self.sigma_trades = QLabel("0")
        self.sigma_trades.setStyleSheet("font-weight: 700;")
        grid.addWidget(self.sigma_trades, 3, 0)
        
        grid.addWidget(QLabel("Circuit Breaker"), 2, 1)
        self.sigma_cb = QLabel("OK")
        self.sigma_cb.setStyleSheet("color: #3fb950; font-weight: 700;")
        grid.addWidget(self.sigma_cb, 3, 1)
        
        risk_layout.addLayout(grid)
        bottom_row.addWidget(risk_container, stretch=3)
        
        # Trades Recentes
        trades_container = QWidget()
        trades_container.setObjectName("metric_card")
        tr_layout = QVBoxLayout(trades_container)
        tr_header = QHBoxLayout()
        tr_title = QLabel("📑 Trades Recentes")
        tr_title.setStyleSheet("color: #e6edf3; font-weight: 700; font-size: 13px;")
        self.tr_count = QLabel("0 total")
        tr_header.addWidget(tr_title)
        tr_header.addStretch()
        tr_header.addWidget(self.tr_count)
        tr_layout.addLayout(tr_header)
        
        self.trades_table = QTableWidget(0, 6)
        self.trades_table.setHorizontalHeaderLabels(["HORA", "SÍMBOLO", "DIR", "STAKE", "PNL", "CONF"])
        h_header = self.trades_table.horizontalHeader()
        if h_header is not None:
            h_header.setStretchLastSection(True)
        
        v_header = self.trades_table.verticalHeader()
        if v_header is not None:
            v_header.setVisible(False)
        self.trades_table.setShowGrid(False)
        self.trades_table.setAlternatingRowColors(True)
        self.trades_table.setStyleSheet("""
            QTableWidget { border: none; background: transparent; font-size: 11px; }
            QHeaderView::section { background: transparent; border: none; font-size: 10px; color: #8b949e; }
        """)
        tr_layout.addWidget(self.trades_table)
        bottom_row.addWidget(trades_container, stretch=5)
        
        layout.addLayout(bottom_row)
        
        # ── Logs do Sistema ──
        logs_container = QWidget()
        logs_container.setObjectName("metric_card")
        log_layout = QVBoxLayout(logs_container)
        log_title = QLabel("terminal>_ Log do Sistema")
        log_title.setStyleSheet("color: #e6edf3; font-weight: 700; font-size: 13px;")
        log_layout.addWidget(log_title)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background: #0d1117; color: #8b949e; border: 1px solid #21262d; font-family: 'Cascadia Code', monospace; font-size: 11px;")
        self.log_text.setFixedHeight(120)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(logs_container)

        scroll.setWidget(content)
        main_layout.addWidget(scroll)
        
        # Timer para o relógio
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_clock)
        self.timer.start(1000)

    def _update_clock(self):
        self.clock_label.setText(datetime.now().strftime("%H:%M:%S"))

    def log_message(self, msg: str, level: str = "INFO"):
        color = "#8b949e"
        if level == "WARN": color = "#d29922"
        elif level == "ERROR" or level == "LOSS": color = "#f85149"
        elif level == "SUCCESS" or level == "OK": color = "#3fb950"
        
        html = f"<span style='color: #21262d;'>{datetime.now().strftime('%M:%S')}</span> <span style='color: {color}; font-weight: bold;'>{level}</span> {msg}"
        self.log_text.append(html)

    # ── API Pública ──

    def update_status(self, status: dict):
        balance = float(status.get("balance", 0))
        initial = float(status.get("initial_balance", 0))
        pnl = balance - initial
        win_rate = float(status.get("win_rate", 0))
        total_trades = int(status.get("total_trades", 0))
        consec = int(status.get("consecutive_losses", 0))
        wins = int(status.get("wins", 0))
        losses = int(status.get("losses", 0))
        
        pct = (pnl / initial * 100) if initial else 0.0

        # Row 1
        self._card_balance.update_value(f"${balance:,.2f}", f"{pct:+.2f}%", "#e6edf3", "#f85149" if pct < 0 else "#3fb950")
        self._card_pnl.update_value(f"${pnl:+,.2f}", f"{total_trades} trades", "#3fb950" if pnl >= 0 else "#f85149")
        self._card_winrate.update_value(f"{win_rate:.1%}", f"{wins}W / {losses}L", "#3fb950" if win_rate >= 0.5 else "#f85149")
        self._card_dd.update_value("0.00%", "máx. dia") # TODO: Add DD
        
        # Equity
        self.equity_chart.update_data(balance)
        
        # Risk Grid
        self.sigma_pnl.setText(f"${pnl:+,.2f}")
        self.sigma_pnl.setStyleSheet(f"color: {'#3fb950' if pnl >= 0 else '#f85149'}; font-weight: 700;")
        self.sigma_loss.setText(f"{consec} / 5")
        self.sigma_trades.setText(str(total_trades))
        
    def on_council_status(self, council: dict):
        # oracle
        if council.get("status") == "idle":
            return
            
        last_dec = council.get("last_decision")
        if not last_dec:
            return
            
        conf = float(last_dec.get("confidence", 0.0))
        is_buy = last_dec.get("action") == "BUY"
        self.donut_chart.set_value(conf, is_buy)
        
        action = last_dec.get("action", "NEUTRAL")
        col = "#3fb950" if action == "BUY" else "#f85149" if action == "SELL" else "#8b949e"
        self.oracle_verdict.setText(f"Veredicto Final\n{action}")
        self.oracle_verdict.setStyleSheet(f"color: {col}; font-weight: bold;")
        
        votes = last_dec.get("votes", [])
        for vote in votes:
            agent = vote.get("agent")
            if agent not in self.agent_cards:
                continue # Ignore unknown agents or handle dynamically
            
            c = float(vote.get("confidence", 0.0))
            a = vote.get("action", "NEUTRAL")
            r = vote.get("reasoning", "")
            self.agent_cards[agent].update_state(a, c, r)

    def on_trade_event(self, trade: dict):
        self.trades_table.insertRow(0)
        status = trade.get("status", "OPEN")
        color = QColor("#d29922")
        
        ts = trade.get("ts", "")
        if ts:
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%H:%M:%S")
            except: pass

        def item(t, c=QColor("#8b949e")):
            it = QTableWidgetItem(str(t))
            it.setForeground(c)
            it.setData(Qt.ItemDataRole.UserRole, trade.get("trade_id"))
            return it

        self.trades_table.setItem(0, 0, item(ts))
        self.trades_table.setItem(0, 1, item(trade.get("symbol", "")))
        self.trades_table.setItem(0, 2, item(trade.get("direction", ""), color))
        self.trades_table.setItem(0, 3, item(f"${float(trade.get('stake', 0)):.2f}"))
        self.trades_table.setItem(0, 4, item("Em aberto", color))
        self.trades_table.setItem(0, 5, item(f"{float(trade.get('confidence', 0)):.1%}"))
        
        self.tr_count.setText(f"{self.trades_table.rowCount()} total")
        self.log_message(f"Trade aberto: {trade.get('direction')} {trade.get('symbol')}", "OK")

    def on_trade_closed(self, trade: dict):
        trade_id = trade.get("trade_id")
        pnl = float(trade.get("pnl", 0.0))
        status = trade.get("status", "")
        color = QColor("#3fb950") if status == "WON" else QColor("#f85149") if status == "LOST" else QColor("#8b949e")
        
        for row in range(self.trades_table.rowCount()):
            it = self.trades_table.item(row, 0)
            if it and it.data(Qt.ItemDataRole.UserRole) == trade_id:
                def item(t, c=QColor("#8b949e")):
                    new_it = QTableWidgetItem(str(t))
                    new_it.setForeground(c)
                    new_it.setData(Qt.ItemDataRole.UserRole, trade_id)
                    return new_it
                    
                self.trades_table.setItem(row, 2, item(trade.get("direction", ""), color))
                self.trades_table.setItem(row, 4, item(f"${pnl:+.2f}", color))
                break
                
        self.log_message(f"Trade fechado: {trade.get('direction')} {trade.get('symbol')} -> PnL: {pnl:+.2f}", "SUCCESS" if pnl > 0 else "LOSS" if pnl < 0 else "OK")

    def on_gemini_advice(self, advice: dict):
        pass # Handle in agents now

    def on_tick(self, tick: dict):
        import time
        now = time.time()
        if not hasattr(self, '_tick_count'):
            self._tick_count = 0
            self._last_tps_time = now
            
        self._tick_count += 1
        elapsed = now - self._last_tps_time
        
        # Como estamos simulando via polling, multiplicamos para refletir as 500 velas baixadas.
        if elapsed >= 1.0:
            tps = (self._tick_count * 50) / elapsed
            self._card_tps.update_value(f"{tps:.1f}", "velas/s", "#3fb950")
            self._tick_count = 0
            self._last_tps_time = now
