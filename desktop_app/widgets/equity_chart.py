"""
desktop_app/widgets/equity_chart.py — Gráfico de linha de Equity
"""
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QLinearGradient, QPainterPath, QBrush
from PyQt6.QtWidgets import QWidget


class EquityChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.data_points: list[float] = [0.0]
        self.max_points = 100
        
        self.line_color = QColor("#00f0ff")  # Cyan neon
        self.bg_color = QColor("#161b22")

    def update_data(self, balance: float):
        self.data_points.append(balance)
        if len(self.data_points) > self.max_points:
            self.data_points.pop(0)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = self.rect()
        width = rect.width()
        height = rect.height()
        
        # Draw background
        painter.fillRect(rect, self.bg_color)
        
        if len(self.data_points) < 2:
            return

        min_val = min(self.data_points)
        max_val = max(self.data_points)
        
        # Add some padding to Y axis
        padding_y = height * 0.1
        range_val = (max_val - min_val) if max_val != min_val else 1.0
        
        step_x = width / (self.max_points - 1)
        
        path = QPainterPath()
        fill_path = QPainterPath()
        
        pts = []
        for i, val in enumerate(self.data_points):
            x = width - (len(self.data_points) - 1 - i) * step_x
            y = height - padding_y - ((val - min_val) / range_val) * (height - 2 * padding_y)
            pts.append(QPointF(x, y))
            
        path.moveTo(pts[0])
        fill_path.moveTo(pts[0].x(), height)
        fill_path.lineTo(pts[0])
        
        for i in range(1, len(pts)):
            # Draw lines. A spline is better but linear is faster
            path.lineTo(pts[i])
            fill_path.lineTo(pts[i])
            
        fill_path.lineTo(pts[-1].x(), height)
        fill_path.closeSubpath()
        
        # Gradient fill
        grad = QLinearGradient(0, 0, 0, height)
        grad.setColorAt(0, QColor(0, 240, 255, 60))
        grad.setColorAt(1, QColor(0, 240, 255, 0))
        
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(grad))
        painter.drawPath(fill_path)
        
        # Draw Line
        pen = QPen(self.line_color, 2)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
