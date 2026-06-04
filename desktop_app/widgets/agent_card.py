"""
desktop_app/widgets/agent_card.py — Card horizontal de conselheiro com QProgressBar customizado
"""
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar


class AgentCard(QWidget):
    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.setObjectName("agent_card")
        self.setStyleSheet("""
            QWidget#agent_card {
                background-color: #161b22;
                border: 1px solid #21262d;
                border-radius: 8px;
            }
        """)
        self.setMinimumWidth(250)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(6)
        
        # Header
        header = QHBoxLayout()
        self.name_label = QLabel(name.upper())
        self.name_label.setStyleSheet("color: #58a6ff; font-weight: 700; font-size: 11px;")
        
        self.icon_label = QLabel("🧠")
        
        header.addWidget(self.name_label)
        header.addStretch()
        header.addWidget(self.icon_label)
        layout.addLayout(header)
        
        # Vote
        self.vote_label = QLabel("NEUTRAL")
        self.vote_label.setStyleSheet("color: #8b949e; font-weight: 700; font-size: 13px;")
        layout.addWidget(self.vote_label)
        
        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #21262d;
                border-radius: 2px;
            }
            QProgressBar::chunk {
                background-color: #3fb950;
                border-radius: 2px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Footer
        footer = QHBoxLayout()
        self.conf_label = QLabel("Confiança: 0%")
        self.conf_label.setStyleSheet("color: #8b949e; font-size: 11px;")
        
        self.reasoning_label = QLabel("Ranging")
        self.reasoning_label.setStyleSheet("color: #8b949e; font-size: 11px;")
        self.reasoning_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        
        footer.addWidget(self.conf_label)
        footer.addStretch()
        footer.addWidget(self.reasoning_label)
        layout.addLayout(footer)

    def update_state(self, vote: str, confidence: float, reasoning: str):
        self.progress_bar.setValue(int(confidence * 100))
        self.conf_label.setText(f"Confiança: {int(confidence * 100)}%")
        
        short_reason = reasoning.split("-")[0].strip() if "-" in reasoning else reasoning
        self.reasoning_label.setText(short_reason[:15])
        
        if vote == "BUY":
            self.vote_label.setText("BUY")
            self.vote_label.setStyleSheet("color: #3fb950; font-weight: 700; font-size: 13px;")
            self.progress_bar.setStyleSheet("""
                QProgressBar { background-color: #21262d; border-radius: 2px; }
                QProgressBar::chunk { background-color: #3fb950; border-radius: 2px; }
            """)
        elif vote == "SELL":
            self.vote_label.setText("SELL")
            self.vote_label.setStyleSheet("color: #f85149; font-weight: 700; font-size: 13px;")
            self.progress_bar.setStyleSheet("""
                QProgressBar { background-color: #21262d; border-radius: 2px; }
                QProgressBar::chunk { background-color: #f85149; border-radius: 2px; }
            """)
        else:
            self.vote_label.setText("NEUTRAL")
            self.vote_label.setStyleSheet("color: #8b949e; font-weight: 700; font-size: 13px;")
            self.progress_bar.setStyleSheet("""
                QProgressBar { background-color: #21262d; border-radius: 2px; }
                QProgressBar::chunk { background-color: #8b949e; border-radius: 2px; }
            """)
