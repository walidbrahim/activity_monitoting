"""
PostureWidget & MotionWidget — Animated real-time visualizers.
"""

import math
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QFrame, QSizePolicy
from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QBrush, QFont,
    QLinearGradient, QRadialGradient, QPainterPath
)
import pyqtgraph as pg
from PyQt6.QtWidgets import QGraphicsItem
from config import config

# ─── Colour helpers ──────────────────────────────────────────────────
def _conf_color(confidence: float) -> QColor:
    """Map 0-100 confidence to a red→amber→green colour."""
    if confidence > 70:
        return QColor("#22C55E")   # green
    elif confidence > 40:
        return QColor("#F59E0B")   # amber
    return QColor("#EF4444")       # red

# ─── Posture drawing primitives (Anatomical Pill Shapes) ──────────────
def _draw_head(p: QPainter, cx: float, cy: float, r: float, color: QColor):
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(color))
    p.drawEllipse(QPointF(cx, cy), r, r)

def _draw_limb(p: QPainter, x1, y1, x2, y2, color: QColor, width: float):
    pen = QPen(color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

# ─── Individual posture renderers ───────────────────────────────────
def paint_standing(p: QPainter, rect: QRectF, color: QColor, _phase: float):
    cx = rect.center().x()
    side = rect.width()
    
    # Anatomical proportions
    head_r = side * 0.12
    torso_w = side * 0.22
    arm_w = side * 0.11
    leg_w = side * 0.13

    head_cy = rect.top() + head_r
    _draw_head(p, cx, head_cy, head_r, color)

    neck_y = head_cy + head_r * 0.8
    hip_y = rect.top() + rect.height() * 0.50
    _draw_limb(p, cx, neck_y, cx, hip_y, color, torso_w)

    # arms
    shoulder_y = neck_y + side * 0.05
    arm_dx = torso_w * 0.5 + arm_w * 0.5 + side * 0.02
    hand_y = hip_y - side * 0.05
    _draw_limb(p, cx - arm_dx, shoulder_y, cx - arm_dx, hand_y, color, arm_w)
    _draw_limb(p, cx + arm_dx, shoulder_y, cx + arm_dx, hand_y, color, arm_w)

    # legs
    foot_y = rect.bottom() - leg_w * 0.5
    leg_dx = torso_w * 0.25
    _draw_limb(p, cx - leg_dx, hip_y, cx - leg_dx, foot_y, color, leg_w)
    _draw_limb(p, cx + leg_dx, hip_y, cx + leg_dx, foot_y, color, leg_w)

def paint_sitting(p: QPainter, rect: QRectF, color: QColor, _phase: float):
    cx = rect.center().x()
    side = rect.width()
    
    head_r = side * 0.12
    torso_w = side * 0.22
    arm_w = side * 0.11
    leg_w = side * 0.13

    head_cy = rect.top() + head_r + side * 0.05
    _draw_head(p, cx, head_cy, head_r, color)

    neck_y = head_cy + head_r * 0.8
    hip_y = rect.top() + rect.height() * 0.52
    _draw_limb(p, cx, neck_y, cx, hip_y, color, torso_w)

    # arms resting on lap
    shoulder_y = neck_y + side * 0.05
    arm_dx = torso_w * 0.5 + arm_w * 0.5
    hand_x = cx + side * 0.18
    _draw_limb(p, cx - arm_dx, shoulder_y, cx - side * 0.18, hip_y, color, arm_w)
    _draw_limb(p, cx + arm_dx, shoulder_y, hand_x, hip_y, color, arm_w)

    # legs — knees bent forward
    knee_dx = side * 0.20
    knee_y = hip_y + side * 0.05
    foot_y = rect.bottom() - leg_w * 0.5
    
    # Upper leg
    _draw_limb(p, cx, hip_y, cx - knee_dx, knee_y, color, leg_w)
    _draw_limb(p, cx, hip_y, cx + knee_dx, knee_y, color, leg_w)
    # Lower leg
    _draw_limb(p, cx - knee_dx, knee_y, cx - knee_dx, foot_y, color, leg_w*0.9)
    _draw_limb(p, cx + knee_dx, knee_y, cx + knee_dx, foot_y, color, leg_w*0.9)

    # chair surface hint
    chair_y = hip_y + side * 0.04
    chair_hw = side * 0.35
    pen = QPen(QColor(config.gui_theme.subtext), 2.0, Qt.PenStyle.DashLine)
    p.setPen(pen)
    p.drawLine(QPointF(cx - chair_hw, chair_y), QPointF(cx + chair_hw, chair_y))

def paint_lying(p: QPainter, rect: QRectF, color: QColor, phase: float):
    cy = rect.center().y() + rect.height() * 0.1
    side = rect.width()
    left = rect.left() + side * 0.10
    right = rect.right() - side * 0.10
    
    head_r = side * 0.12
    torso_w = side * 0.20
    arm_w = side * 0.10
    leg_w = side * 0.11

    # Breathing micro-animation: body rises slightly
    breath = math.sin(phase) * side * 0.02

    head_cx = left + head_r
    _draw_head(p, head_cx, cy, head_r, color)

    neck_x = head_cx + head_r * 0.6
    hip_x = left + (right - left) * 0.48
    _draw_limb(p, neck_x, cy - breath, hip_x, cy, color, torso_w + breath*1.5)

    # arms draping
    shoulder_x = neck_x + side * 0.05
    arm_end_x = shoulder_x + side * 0.15
    arm_dy = side * 0.12
    _draw_limb(p, shoulder_x, cy, arm_end_x, cy + arm_dy + breath, color, arm_w)

    # legs
    foot_x = right - leg_w * 0.5
    _draw_limb(p, hip_x, cy, foot_x, cy, color, leg_w)

    # bed surface hint
    bed_y = cy + torso_w * 0.5 + side * 0.02
    pen = QPen(QColor(config.gui_theme.subtext), 2.0, Qt.PenStyle.DashLine)
    p.setPen(pen)
    p.drawLine(QPointF(rect.left(), bed_y), QPointF(rect.right(), bed_y))

def paint_fallen(p: QPainter, rect: QRectF, color: QColor, _phase: float):
    cx = rect.center().x()
    cy = rect.center().y() + rect.height() * 0.15
    side = rect.width()
    
    head_r = side * 0.12
    torso_w = side * 0.20
    arm_w = side * 0.10
    leg_w = side * 0.11

    angle = math.radians(-65)
    body_len = side * 0.35
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    head_cx = cx - body_len * cos_a * 0.8
    head_cy = cy + body_len * sin_a * 0.8
    _draw_head(p, head_cx, head_cy, head_r, color)

    neck_x = head_cx + head_r * cos_a
    neck_y = head_cy - head_r * sin_a
    hip_x = neck_x + body_len * cos_a
    hip_y = neck_y - body_len * sin_a
    _draw_limb(p, neck_x, neck_y, hip_x, hip_y, color, torso_w)

    # arms sprawled
    arm_len = side * 0.25
    _draw_limb(p, neck_x + side*0.05, neck_y, neck_x + arm_len * 0.8, neck_y + arm_len * 0.6, color, arm_w)
    _draw_limb(p, neck_x + side*0.05, neck_y, neck_x - arm_len * 0.4, neck_y - arm_len * 0.7, color, arm_w)

    # legs tangled
    leg_len = side * 0.30
    _draw_limb(p, hip_x, hip_y, hip_x + leg_len, hip_y + side*0.05, color, leg_w)
    _draw_limb(p, hip_x, hip_y, hip_x + leg_len * 0.6, hip_y - leg_len * 0.8, color, leg_w)

    # floor line
    ground_y = rect.bottom() - side * 0.05
    pen = QPen(QColor(config.gui_theme.subtext), 2.0, Qt.PenStyle.DashLine)
    p.setPen(pen)
    p.drawLine(QPointF(rect.left() + side*0.1, ground_y), QPointF(rect.right() - side*0.1, ground_y))

def paint_unknown(p: QPainter, rect: QRectF, color: QColor, _phase: float):
    cx = rect.center().x()
    cy = rect.center().y()
    side = rect.width()

    faded = QColor(color)
    faded.setAlpha(40)
    
    head_r = side * 0.12
    head_cy = rect.top() + side * 0.25
    _draw_head(p, cx, head_cy, head_r, faded)
    _draw_limb(p, cx, head_cy + head_r * 0.8, cx, cy + side * 0.2, faded, side*0.18)

    font = QFont("Arial", int(side * 0.35), QFont.Weight.Bold)
    p.setFont(font)
    p.setPen(QPen(QColor(config.gui_theme.subtext), 1))
    p.drawText(rect, Qt.AlignmentFlag.AlignCenter, "?")

POSTURE_PAINTERS = {
    "Standing":   paint_standing,
    "Sitting":    paint_sitting,
    "Lying Down": paint_lying,
    "Fallen":     paint_fallen,
    "Unknown":    paint_unknown,
    "":           paint_unknown,
}

# ─── Posture Widget ──────────────────────────────────────────────────
class PostureWidget(QWidget):
    TRANSITION_MS = 350
    TICK_MS = 33

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(60, 60)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._posture = "Unknown"
        self._prev_posture = "Unknown"
        self._confidence = 0.0
        self._transition_t = 1.0
        self._transition_step = self.TICK_MS / self.TRANSITION_MS
        self._phase = 0.0
        self._phase_speed = 0.06

        self._timer = QTimer(self)
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def set_posture(self, posture: str, confidence: float = 0.0):
        if posture != self._posture:
            self._prev_posture = self._posture
            self._posture = posture
            self._transition_t = 0.0
        self._confidence = confidence

    def _tick(self):
        self._phase = (self._phase + self._phase_speed) % (2 * math.pi)
        if self._transition_t < 1.0:
            self._transition_t = min(1.0, self._transition_t + self._transition_step)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w, h = self.width(), self.height()
        side = min(w, h)
        draw_rect = QRectF((w - side) / 2, (h - side) / 2, side, side)
        figure_rect = draw_rect.adjusted(side * 0.10, side * 0.10, -side * 0.10, -side * 0.10)

        glow_color = _conf_color(self._confidence) if self._posture != "Fallen" else QColor("#EF4444")
        
        # Glow
        grad = QRadialGradient(draw_rect.center(), side * 0.45)
        gc = QColor(glow_color)
        pulse = 0.5 + 0.5*math.sin(self._phase*3) if self._posture == "Fallen" else 0.0
        gc.setAlpha(int(30 + pulse * 50))
        grad.setColorAt(0.0, gc)
        grad.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(grad))
        p.drawEllipse(draw_rect.center(), side * 0.44, side * 0.44)

        # Draw figure
        base_color = QColor(config.gui_theme.text) if self._posture != "Fallen" else QColor("#EF4444")

        if self._transition_t < 1.0 and self._prev_posture != self._posture:
            prev_painter = POSTURE_PAINTERS.get(self._prev_posture, paint_unknown)
            curr_painter = POSTURE_PAINTERS.get(self._posture, paint_unknown)

            prev_color = QColor(config.gui_theme.text)
            prev_color.setAlpha(int(255 * (1.0 - self._transition_t)))
            curr_color = QColor(base_color)
            curr_color.setAlpha(int(255 * self._transition_t))

            prev_painter(p, figure_rect, prev_color, self._phase)
            curr_painter(p, figure_rect, curr_color, self._phase)
        else:
            painter_fn = POSTURE_PAINTERS.get(self._posture, paint_unknown)
            painter_fn(p, figure_rect, base_color, self._phase)
            
        p.end()

# ─── Motion Widget ───────────────────────────────────────────────────
class MotionWidget(QWidget):
    TICK_MS = 33

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(60, 60)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._motion = "Unknown"
        self._phase = 0.0
        self._phase_speed = 0.1

        self._timer = QTimer(self)
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def set_motion(self, motion: str):
        if not motion: motion = "Unknown"
        if ("Breathing" in motion) or ("Resting" in motion):
            self._motion = "Breathing"
            self._phase_speed = 0.05
        elif "Major" in motion:
            self._motion = "Major"
            self._phase_speed = 0.5
        elif "Restless" in motion or "Shift" in motion:
            self._motion = "Restless"
            self._phase_speed = 0.2
        elif "Static" in motion:
            self._motion = "Static"
            self._phase_speed = 0.0
        else:
            self._motion = "Unknown"

    def _tick(self):
        self._phase = (self._phase + self._phase_speed) % (2 * math.pi)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()
        side = min(w, h)
        cx, cy = w / 2, h / 2

        color = QColor(config.gui_theme.occupant)
        
        if self._motion == "Breathing":
            # Slow pulsing concentric circles
            p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 100), 2.0))
            r1 = side * 0.15 + math.sin(self._phase) * side * 0.05
            r2 = side * 0.25 + math.sin(self._phase - 1.0) * side * 0.08
            p.drawEllipse(QPointF(cx, cy), r1, r1)
            p.drawEllipse(QPointF(cx, cy), r2, r2)
            
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(color))
            p.drawEllipse(QPointF(cx, cy), side * 0.08, side * 0.08)

        elif self._motion == "Restless":
            # Erratic zig-zags
            p.setPen(QPen(color, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            path = QPainterPath()
            path.moveTo(cx - side*0.3, cy)
            for i in range(1, 7):
                dx = cx - side*0.3 + (side*0.6) * (i / 6.0)
                dy = cy + math.sin(self._phase * i * 1.5) * side * 0.15
                path.lineTo(dx, dy)
            p.drawPath(path)

        elif self._motion == "Major":
            # Dynamic speed lines / blur
            p.setPen(QPen(QColor("#F59E0B"), 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            for i in range(5):
                angle = self._phase + i * (2*math.pi/5)
                r_in = side * 0.15
                r_out = side * 0.35 + math.sin(self._phase*5 + i) * side * 0.1
                x1, y1 = cx + math.cos(angle)*r_in, cy + math.sin(angle)*r_in
                x2, y2 = cx + math.cos(angle)*r_out, cy + math.sin(angle)*r_out
                p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
            
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor("#F59E0B")))
            p.drawEllipse(QPointF(cx, cy), side * 0.15, side * 0.15)

        elif self._motion == "Static":
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(config.gui_theme.subtext)))
            p.drawEllipse(QPointF(cx, cy), side * 0.1, side * 0.1)
            pen = QPen(QColor(config.gui_theme.subtext), 2.0, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawLine(QPointF(cx - side*0.3, cy), QPointF(cx + side*0.3, cy))

        else:
            p.setPen(QPen(QColor(config.gui_theme.subtext), 1.0))
            p.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, "-")

        p.end()

# ─── Composite Card ─────────────────────────────────────────────────
class PostureCard(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("PostureCard")
        self.setStyleSheet(f"""
            QFrame#PostureCard {{
                background-color: {config.gui_theme.card_bg};
                border-radius: 10px;
                padding: 10px;
                border: 1px solid rgba(255, 255, 255, 0.08);
            }}
            QLabel {{
                background-color: transparent;
            }}
        """)

        # Standard proportional layout matching other cards
        root = QHBoxLayout(self)
        root.setContentsMargins(15, 15, 15, 15)
        root.setSpacing(15)

        # Left: Vertical graphics (Posture + Motion)
        graphic_layout = QVBoxLayout()
        
        self.posture_widget = PostureWidget()
        self.posture_widget.setMinimumSize(95, 95)
        
        self.motion_widget = MotionWidget()
        self.motion_widget.setMinimumSize(95, 95)

        graphic_layout.addWidget(self.posture_widget)
        graphic_layout.addWidget(self.motion_widget)
        
        # Right: Text Column matching CardWidget exactly
        text_layout = QVBoxLayout()
        
        title = QLabel("🧍 Posture & Motion")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {config.gui_theme.text};")
        text_layout.addWidget(title)
        
        self._labels: dict[str, QLabel] = {}
        for key in ("Posture", "Posture conf.", "Height (Range)", "Motion"):
            row = QHBoxLayout()
            lbl_key = QLabel(key)
            lbl_key.setStyleSheet(f"color: {config.gui_theme.subtext}; font-size: 20px;")
            
            lbl_val = QLabel("--")
            lbl_val.setStyleSheet(f"color: {config.gui_theme.text}; font-size: 20px; font-weight: bold;")
            lbl_val.setAlignment(Qt.AlignmentFlag.AlignRight)
            
            row.addWidget(lbl_key)
            row.addWidget(lbl_val)
            text_layout.addLayout(row)
            self._labels[key] = lbl_val

        root.addLayout(graphic_layout, stretch=1)
        root.addLayout(text_layout, stretch=2)

    def update_values(self, **kwargs):
        for k, v in kwargs.items():
            if k in self._labels:
                self._labels[k].setText(str(v))

    def set_posture_state(self, posture: str, confidence: float, motion: str = ""):
        self.posture_widget.set_posture(posture, confidence)
        self.motion_widget.set_motion(motion)

    def set_color(self, hex_color):
        self.setStyleSheet(f"""
            QFrame#PostureCard {{
                background-color: {hex_color};
                border-radius: 10px;
                padding: 10px;
                border: 1px solid rgba(255, 255, 255, 0.12);
            }}
            QLabel {{
                background-color: transparent;
            }}
        """)

# ─── Radar Posture Map Item ──────────────────────────────────────────
class RadarPostureItem(pg.GraphicsObject):
    """
    A custom Pyqtgraph item that renders the same posture anatomies on the radar map.
    It ignores view transformations so it maintains a clean crisp pixel size 
    regardless of how the user zooms the room.
    """
    def __init__(self):
        super().__init__()
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self._posture = "Unknown"
        self._confidence = 0.0
        self._phase = 0.0
        self._icon_size = 40.0 # crisp fixed pixel size
        self.setZValue(100) # Draw on top of other radar dots

        # Built-in animation loop
        self._timer = QTimer()
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def set_state(self, posture: str, confidence: float):
        if posture != self._posture or confidence != self._confidence:
            self._posture = posture
            self._confidence = confidence
            self.update()

    def _tick(self):
        self._phase = (self._phase + 0.06) % (2 * math.pi)
        # We only really need to force update if an animation is moving (breathing, fallen pulse)
        if self._posture in ["Lying Down", "Fallen", "Unknown"]:
            self.update()

    def boundingRect(self):
        # The bounding rect is in local coordinate space (pixels because of ignores-transform)
        half = self._icon_size / 2.0
        return QRectF(-half, -half, self._icon_size, self._icon_size)

    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        
        # Center the rect on 0,0 since we move the item via setPos()
        half = self._icon_size / 2.0
        draw_rect = QRectF(-half, -half, self._icon_size, self._icon_size)

        # Draw a semi-transparent dark circle background for contrast
        painter.setPen(Qt.PenStyle.NoPen)
        bg = QColor(config.gui_theme.fig_bg)
        bg.setAlpha(180)
        painter.setBrush(QBrush(bg))
        painter.drawEllipse(draw_rect.center(), half, half)

        # Decide color
        base_color = QColor(config.gui_theme.text) if self._posture != "Fallen" else QColor("#EF4444")
        if self._confidence < 30 and self._posture != "Fallen":
            base_color = QColor(config.gui_theme.subtext)

        # Draw the anatomic shape scaled perfectly for the icon size
        painter_fn = POSTURE_PAINTERS.get(self._posture, paint_unknown)
        painter_fn(painter, draw_rect.adjusted(4, 4, -4, -4), base_color, self._phase)
