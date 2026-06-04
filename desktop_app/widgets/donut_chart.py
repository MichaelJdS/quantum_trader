"""
desktop_app/widgets/donut_chart.py — Gráfico de Rosca animado via QPainter
"""
from PyQt6.QtCore import Qt, QRectF, QPropertyAnimation, pyqtProperty
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QPainterPath
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel


class DonutChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(140, 140)
        self._confidence = 0.0
        self._target_confidence = 0.0
        
        self.color_track = QColor("#21262d")
        self.color_fill = QColor("#3fb950")  # Default green
        self.thickness = 12

        # Animation
        self.anim = QPropertyAnimation(self, b"animValue")
        self.anim.setDuration(800)

    @pyqtProperty(float)
    def animValue(self):
        return self._confidence

    @animValue.setter
    def animValue(self, val):
        self._confidence = val
        self.update()

    def set_value(self, confidence: float, is_buy: bool):
        self._target_confidence = max(0.0, min(1.0, confidence))
        self.color_fill = QColor("#3fb950") if is_buy else QColor("#f85149")
        
        self.anim.stop()
        self.anim.setStartValue(self._confidence)
        self.anim.setEndValue(self._target_confidence)
        self.anim.start()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        width = rect.width()
        height = rect.height()
        size = min(width, height) - self.thickness * 2
        
        x = (width - size) / 2
        y = (height - size) / 2
        draw_rect = QRectF(x, y, size, size)

        # Draw track
        pen_track = QPen(self.color_track, self.thickness)
        pen_track.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_track)
        painter.drawArc(draw_rect, 0, 360 * 16)

        # Draw fill
        pen_fill = QPen(self.color_fill, self.thickness)
        pen_fill.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_fill)
        
        # Qt drawArc uses 1/16th of a degree. 90 * 16 is 12 o'clock.
        start_angle = 90 * 16
        span_angle = -int(self._confidence * 360 * 16)
        painter.drawArc(draw_rect, start_angle, span_angle)

        # Draw text inside
        font = QFont("Inter", 24, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QColor("#e6edf3"))
        text = f"{int(self._confidence * 100)}%"
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        
        font_sub = QFont("Inter", 10, QFont.Weight.Medium)
        painter.setFont(font_sub)
        painter.setPen(QColor("#8b949e"))
        painter.drawText(QRectF(x, y + 25, size, size), Qt.AlignmentFlag.AlignCenter, "CONF")
