import sys
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import pyqtgraph as pg
from datetime import datetime

class TradingDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QuantumTrader Core v3.0 | Institutional")
        self.resize(1500, 950)
        self.setStyleSheet("QMainWindow{background:#0b0f19} *{color:#e6edf3;font-family:'JetBrains Mono',monospace}")
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout(central)

        # 📈 Gráfico Principal
        self.chart = pg.PlotWidget(background="#0b0f19")
        self.chart.showGrid(x=True, y=True, alpha=0.2)
        self.chart.setTitle("Market Feed (R_100)", color="#58a6ff")
        self.curve = self.chart.plot(pen=pg.mkPen("#58a6ff", width=2))
        grid.addWidget(self.chart, 0, 0, 3, 2)

        # 🤖 Painel de Agentes
        agents_box = QGroupBox("🤖 15 AI Agents Status")
        agents_layout = QVBoxLayout()
        self.agent_labels = {f"a{i}": QLabel(f"Agent {i+1}: Aguardando dados...") for i in range(15)}
        for lbl in self.agent_labels.values():
            lbl.setStyleSheet("padding:3px; border-bottom: 1px solid #21262d; font-size: 12px;")
            agents_layout.addWidget(lbl)
        agents_box.setLayout(agents_layout)
        grid.addWidget(agents_box, 0, 2, 2, 1)

        # 📊 Métricas & Risco
        metrics = QGroupBox("📊 Risk & Performance")
        m_layout = QVBoxLayout()
        self.lbl_signal = QLabel("Signal: HOLD")
        self.lbl_signal.setStyleSheet("font-size:16px; color:#ffd700; font-weight:bold;")
        self.lbl_dd = QLabel("Drawdown: 0.00%")
        self.lbl_pnl = QLabel("PnL: $0.00")
        self.lbl_trades = QLabel("Trades: 0 | WR: 0.0%")
        for lbl in [self.lbl_signal, self.lbl_dd, self.lbl_pnl, self.lbl_trades]:
            m_layout.addWidget(lbl)
        metrics.setLayout(m_layout)
        grid.addWidget(metrics, 2, 2, 1, 1)

        # 🎛️ Botão de Controle
        btn_box = QGroupBox("🎛️ System Control")
        b_layout = QHBoxLayout()
        self.btn_start = QPushButton("▶ INICIAR SISTEMA")
        self.btn_start.setStyleSheet("padding:10px; background:#238636; border-radius:6px; font-weight:bold; font-size:14px;")
        b_layout.addWidget(self.btn_start)
        btn_box.setLayout(b_layout)
        grid.addWidget(btn_box, 3, 0, 1, 3)

        # 📜 Console de Logs
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("background:#161b22; color:#8b949e; font-size:12px; font-family:'Consolas',monospace;")
        grid.addWidget(self.log, 4, 0, 1, 3)

    def log_msg(self, msg: str):
        """Recebe logs da thread async via pyqtSignal (thread-safe)"""
        self.log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def update_data(self, prices, signal, score, votes, dd, pnl, trades, wr):
        """Recebe dados da thread async via pyqtSignal (thread-safe)"""
        if prices:
            self.curve.setData(prices[-300:])
            
        self.lbl_signal.setText(f"Signal: {signal} | Conf: {score:.2%}")
        color = "#3fb950" if signal == "CALL" else "#f85149" if signal == "PUT" else "#8b949e"
        self.lbl_signal.setStyleSheet(f"font-size:16px; color:{color}; font-weight:bold;")
        
        self.lbl_dd.setText(f"Drawdown: {dd:.2f}%")
        self.lbl_dd.setStyleSheet("color:#f85149; font-weight:bold;" if dd > 3 else "")
        
        self.lbl_pnl.setText(f"PnL: ${pnl:.2f}")
        self.lbl_trades.setText(f"Trades: {trades} | WR: {wr:.1f}%")
        
        for i, (k, v) in enumerate(votes.items()):
            if i < 15 and f"a{i}" in self.agent_labels:
                self.agent_labels[f"a{i}"].setText(f"✅ {k}: {v['signal']} ({v['conf']:.2f})")