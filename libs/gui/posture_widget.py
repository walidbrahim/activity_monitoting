"""
PostureWidget & MotionWidget — Animated real-time visualizers.

Motion is collapsed to three display classes here in the visualizer only:
  "Resting"  — Still, Breathing, Static, Monitoring, Apnea, weak signal
  "Moving"   — active body motion in any zone (non-walking)
  "Walking"  — pipeline motion_str == 'Walking' (net XY translation in transit)
The pipeline is NOT changed.
"""

import math
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QSizePolicy
from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QBrush, QFont,
    QRadialGradient, QPainterPath
)
import pyqtgraph as pg
from PyQt6.QtWidgets import QGraphicsItem
from config import config

# ─── Colour helpers ──────────────────────────────────────────────────
def _conf_color(confidence: float) -> QColor:
    """Map 0-100 confidence to a red→amber→green colour."""
    if confidence > 70:
        return QColor("#22C55E")
    elif confidence > 40:
        return QColor("#F59E0B")
    return QColor("#EF4444")


def _draw_progress_ring(p: QPainter, draw_rect: QRectF, fill_pct: float,
                         ring_color: QColor, thickness: float):
    """
    Draw a progress arc ring from 12-o'clock sweeping clockwise.
      fill_pct  — 0–100 (what fraction of the circle to fill)
      Background (dim full circle) + colored sweep arc.
    """
    inset = draw_rect.adjusted(thickness / 2, thickness / 2,
                                -thickness / 2, -thickness / 2)
    # Background ring
    bg = QColor(ring_color)
    bg.setAlpha(35)
    p.setPen(QPen(bg, thickness, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(inset)

    # Colored sweep (clockwise from top = negative span in Qt)
    sweep = int(fill_pct / 100.0 * 360.0 * 16)
    if sweep <= 0:
        return
    p.setPen(QPen(ring_color, thickness, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    p.drawArc(inset, 90 * 16, -sweep)


# ─── Posture drawing primitives ──────────────────────────────────────
def _draw_head(p: QPainter, cx: float, cy: float, r: float, color: QColor):
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(color))
    p.drawEllipse(QPointF(cx, cy), r, r)


def _draw_limb(p: QPainter, x1, y1, x2, y2, color: QColor, width: float):
    pen = QPen(color, width, Qt.PenStyle.SolidLine,
               Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.drawLine(QPointF(x1, y1), QPointF(x2, y2))


# ─── Individual posture renderers ────────────────────────────────────
def paint_standing(p: QPainter, rect: QRectF, color: QColor, _phase: float):
    cx   = rect.center().x()
    side = rect.width()

    head_r  = side * 0.12
    torso_w = side * 0.22
    arm_w   = side * 0.11
    leg_w   = side * 0.13

    head_cy = rect.top() + head_r
    _draw_head(p, cx, head_cy, head_r, color)

    neck_y = head_cy + head_r * 0.8
    hip_y  = rect.top() + rect.height() * 0.50
    _draw_limb(p, cx, neck_y, cx, hip_y, color, torso_w)

    # Arms angled slightly away from body (more lifelike)
    shoulder_y = neck_y + side * 0.05
    arm_dx     = torso_w * 0.5 + arm_w * 0.5 + side * 0.02
    hand_dx    = arm_dx + side * 0.07   # flare outward at hand
    hand_y     = hip_y - side * 0.04
    _draw_limb(p, cx - arm_dx, shoulder_y, cx - hand_dx, hand_y, color, arm_w)
    _draw_limb(p, cx + arm_dx, shoulder_y, cx + hand_dx, hand_y, color, arm_w)

    # Legs
    foot_y = rect.bottom() - leg_w * 0.5
    leg_dx = torso_w * 0.25
    _draw_limb(p, cx - leg_dx, hip_y, cx - leg_dx, foot_y, color, leg_w)
    _draw_limb(p, cx + leg_dx, hip_y, cx + leg_dx, foot_y, color, leg_w)


def paint_sitting(p: QPainter, rect: QRectF, color: QColor, _phase: float):
    cx   = rect.center().x()
    side = rect.width()

    head_r  = side * 0.12
    torso_w = side * 0.22
    arm_w   = side * 0.11
    leg_w   = side * 0.13

    head_cy = rect.top() + head_r + side * 0.05
    _draw_head(p, cx, head_cy, head_r, color)

    neck_y = head_cy + head_r * 0.8
    hip_y  = rect.top() + rect.height() * 0.52
    _draw_limb(p, cx, neck_y, cx, hip_y, color, torso_w)

    # Chair back — subtle vertical rectangle behind the torso
    back_x = cx + torso_w * 0.6
    cb_color = QColor(config.gui_theme.subtext)
    cb_color.setAlpha(100)
    _draw_limb(p, back_x, neck_y - side * 0.05, back_x, hip_y + side * 0.04,
               cb_color, side * 0.04)

    # Arms resting on lap
    shoulder_y = neck_y + side * 0.05
    arm_dx     = torso_w * 0.5 + arm_w * 0.5
    _draw_limb(p, cx - arm_dx, shoulder_y, cx - side * 0.18, hip_y, color, arm_w)
    _draw_limb(p, cx + arm_dx, shoulder_y, cx + side * 0.18, hip_y, color, arm_w)

    # Legs — knees bent forward
    knee_dx = side * 0.20
    knee_y  = hip_y + side * 0.05
    foot_y  = rect.bottom() - leg_w * 0.5
    _draw_limb(p, cx, hip_y, cx - knee_dx, knee_y, color, leg_w)
    _draw_limb(p, cx, hip_y, cx + knee_dx, knee_y, color, leg_w)
    _draw_limb(p, cx - knee_dx, knee_y, cx - knee_dx, foot_y, color, leg_w * 0.9)
    _draw_limb(p, cx + knee_dx, knee_y, cx + knee_dx, foot_y, color, leg_w * 0.9)

    # Chair seat hint
    chair_y  = hip_y + side * 0.04
    chair_hw = side * 0.35
    p.setPen(QPen(QColor(config.gui_theme.subtext), 2.0, Qt.PenStyle.DashLine))
    p.drawLine(QPointF(cx - chair_hw, chair_y), QPointF(cx + chair_hw, chair_y))


def paint_lying(p: QPainter, rect: QRectF, color: QColor, phase: float):
    cy   = rect.center().y() + rect.height() * 0.1
    side = rect.width()
    left = rect.left()  + side * 0.10
    right = rect.right() - side * 0.10

    head_r  = side * 0.12
    torso_w = side * 0.20
    arm_w   = side * 0.10
    leg_w   = side * 0.11

    breath = math.sin(phase) * side * 0.02

    # Pillow hint (filled oval behind / under the head)
    pillow_cx = left + head_r * 0.35
    pillow_c  = QColor(config.gui_theme.subtext)
    pillow_c.setAlpha(70)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(pillow_c))
    p.drawEllipse(QPointF(pillow_cx, cy), head_r * 1.35, head_r * 0.85)

    head_cx = left + head_r
    _draw_head(p, head_cx, cy, head_r, color)

    neck_x = head_cx + head_r * 0.6
    hip_x  = left + (right - left) * 0.48
    _draw_limb(p, neck_x, cy - breath, hip_x, cy, color, torso_w + breath * 1.5)

    # Arm draping
    shoulder_x = neck_x + side * 0.05
    arm_end_x  = shoulder_x + side * 0.15
    _draw_limb(p, shoulder_x, cy, arm_end_x, cy + side * 0.12 + breath, color, arm_w)

    # Legs
    foot_x = right - leg_w * 0.5
    _draw_limb(p, hip_x, cy, foot_x, cy, color, leg_w)

    # Bed surface
    bed_y = cy + torso_w * 0.5 + side * 0.02
    p.setPen(QPen(QColor(config.gui_theme.subtext), 2.0, Qt.PenStyle.DashLine))
    p.drawLine(QPointF(rect.left(), bed_y), QPointF(rect.right(), bed_y))


def paint_fallen(p: QPainter, rect: QRectF, color: QColor, _phase: float):
    cx   = rect.center().x()
    cy   = rect.center().y() + rect.height() * 0.15
    side = rect.width()

    head_r  = side * 0.12
    torso_w = side * 0.20
    arm_w   = side * 0.10
    leg_w   = side * 0.11

    angle   = math.radians(-65)
    body_len = side * 0.35
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    head_cx = cx - body_len * cos_a * 0.8
    head_cy = cy + body_len * sin_a * 0.8
    _draw_head(p, head_cx, head_cy, head_r, color)

    neck_x = head_cx + head_r * cos_a
    neck_y = head_cy - head_r * sin_a
    hip_x  = neck_x + body_len * cos_a
    hip_y  = neck_y - body_len * sin_a
    _draw_limb(p, neck_x, neck_y, hip_x, hip_y, color, torso_w)

    arm_len = side * 0.25
    _draw_limb(p, neck_x + side * 0.05, neck_y,
               neck_x + arm_len * 0.8,  neck_y + arm_len * 0.6, color, arm_w)
    _draw_limb(p, neck_x + side * 0.05, neck_y,
               neck_x - arm_len * 0.4,  neck_y - arm_len * 0.7, color, arm_w)

    leg_len = side * 0.30
    _draw_limb(p, hip_x, hip_y, hip_x + leg_len,       hip_y + side * 0.05, color, leg_w)
    _draw_limb(p, hip_x, hip_y, hip_x + leg_len * 0.6, hip_y - leg_len * 0.8, color, leg_w)

    ground_y = rect.bottom() - side * 0.05
    p.setPen(QPen(QColor(config.gui_theme.subtext), 2.0, Qt.PenStyle.DashLine))
    p.drawLine(QPointF(rect.left() + side * 0.1, ground_y),
               QPointF(rect.right() - side * 0.1, ground_y))

    # Pulsing "!" alert above figure
    pulse      = 0.5 + 0.5 * math.sin(_phase * 3)
    alert_col  = QColor("#EF4444")
    alert_col.setAlpha(int(180 + pulse * 75))
    font = QFont("Arial", max(8, int(side * 0.24)), QFont.Weight.Bold)
    p.setFont(font)
    p.setPen(QPen(alert_col, 1))
    p.drawText(QRectF(rect.right() - side * 0.35, rect.top(),
                      side * 0.32, side * 0.32),
               Qt.AlignmentFlag.AlignCenter, "!")


def paint_unknown(p: QPainter, rect: QRectF, color: QColor, _phase: float):
    cx   = rect.center().x()
    cy   = rect.center().y()
    side = rect.width()

    # Faded dashed ring
    ring_c = QColor(config.gui_theme.subtext)
    ring_c.setAlpha(55)
    p.setPen(QPen(ring_c, side * 0.05, Qt.PenStyle.DashLine))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QPointF(cx, cy), side * 0.35, side * 0.35)

    font = QFont("Arial", int(side * 0.36), QFont.Weight.Bold)
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

# ─── Motion classification (visualizer-only, no pipeline changes) ─────
_RESTING_KEYWORDS = ("breath", "rest", "still", "static", "monitor", "apnea", "weak", "room")

def _classify_motion(motion_str: str) -> tuple:
    """
    Collapse all pipeline motion strings to three display classes.
    Returns (display_class: str, ring_intensity_pct: float).
      "Walking" / 95.0  — net spatial translation in transit
      "Resting" / 30.0  — calm resting states
      "Moving"  / 85.0  — any active non-walking motion
      "Unknown" / 0.0   — no data
    """
    s = (motion_str or "").lower().strip()
    if not s or s == "unknown":
        return "Unknown", 0.0
    if s == "walking":
        return "Walking", 95.0
    for kw in _RESTING_KEYWORDS:
        if kw in s:
            return "Resting", 30.0
    return "Moving", 85.0


# ─── Posture Widget ──────────────────────────────────────────────────
class PostureWidget(QWidget):
    TRANSITION_MS = 350
    TICK_MS       = 33

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(60, 60)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._posture      = "Unknown"
        self._prev_posture = "Unknown"
        self._confidence   = 0.0
        self._transition_t = 1.0
        self._transition_step = self.TICK_MS / self.TRANSITION_MS
        self._phase        = 0.0
        self._phase_speed  = 0.06

        self._timer = QTimer(self)
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def set_posture(self, posture: str, confidence: float = 0.0):
        if posture != self._posture:
            self._prev_posture = self._posture
            self._posture      = posture
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

        ring_thick  = max(4.0, side * 0.055)
        margin      = ring_thick + side * 0.04
        draw_rect   = QRectF((w - side) / 2, (h - side) / 2, side, side)
        figure_rect = draw_rect.adjusted(margin, margin, -margin, -margin)

        glow_color = (_conf_color(self._confidence)
                      if self._posture != "Fallen" else QColor("#EF4444"))

        # Radial glow
        grad = QRadialGradient(draw_rect.center(), side * 0.45)
        gc   = QColor(glow_color)
        pulse = 0.5 + 0.5 * math.sin(self._phase * 3) if self._posture == "Fallen" else 0.0
        gc.setAlpha(int(30 + pulse * 50))
        grad.setColorAt(0.0, gc)
        grad.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(grad))
        p.drawEllipse(draw_rect.center(), side * 0.44, side * 0.44)

        # Confidence progress ring
        ring_color = (_conf_color(self._confidence)
                      if self._posture != "Fallen" else QColor("#EF4444"))
        _draw_progress_ring(p, draw_rect, self._confidence, ring_color, ring_thick)

        # Posture figure (with cross-fade transition)
        base_color = (QColor(config.gui_theme.text)
                      if self._posture != "Fallen" else QColor("#EF4444"))
        if self._transition_t < 1.0 and self._prev_posture != self._posture:
            prev_fn = POSTURE_PAINTERS.get(self._prev_posture, paint_unknown)
            curr_fn = POSTURE_PAINTERS.get(self._posture,      paint_unknown)
            pc = QColor(config.gui_theme.text)
            pc.setAlpha(int(255 * (1.0 - self._transition_t)))
            cc = QColor(base_color)
            cc.setAlpha(int(255 * self._transition_t))
            prev_fn(p, figure_rect, pc, self._phase)
            curr_fn(p, figure_rect, cc, self._phase)
        else:
            POSTURE_PAINTERS.get(self._posture, paint_unknown)(
                p, figure_rect, base_color, self._phase)

        p.end()


# ─── Motion Widget ───────────────────────────────────────────────────
class MotionWidget(QWidget):
    TICK_MS = 33

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(60, 60)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._motion      = "Unknown"   # "Resting" | "Moving" | "Unknown"
        self._intensity   = 0.0        # ring fill %
        self._phase       = 0.0
        self._phase_speed = 0.05

        self._timer = QTimer(self)
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def set_motion(self, motion: str):
        cls, intensity      = _classify_motion(motion)
        self._motion        = cls
        self._intensity     = intensity
        if cls == "Resting":
            self._phase_speed = 0.05
        elif cls == "Walking":
            self._phase_speed = 0.20
        else:
            self._phase_speed = 0.35

    def _tick(self):
        self._phase = (self._phase + self._phase_speed) % (2 * math.pi)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w, h = self.width(), self.height()
        side = min(w, h)
        cx, cy = w / 2, h / 2

        ring_thick = max(4.0, side * 0.055)
        draw_rect  = QRectF((w - side) / 2, (h - side) / 2, side, side)
        margin     = ring_thick + side * 0.04
        inner_r    = (side - 2 * margin) / 2

        motion_color = (QColor("#22C55E") if self._motion == "Resting"  else
                        QColor("#14B8A6") if self._motion == "Walking"  else
                        QColor("#F59E0B") if self._motion == "Moving"   else
                        QColor(config.gui_theme.subtext))

        # Motion-intensity progress ring
        _draw_progress_ring(p, draw_rect, self._intensity, motion_color, ring_thick)

        if self._motion == "Resting":
            # Slow pulsing concentric rings — calm breathing visual
            for base_r, offset in [(0.35, 0.0), (0.60, -1.0)]:
                r = inner_r * base_r + math.sin(self._phase + offset) * inner_r * 0.10
                c = QColor(motion_color)
                c.setAlpha(110 if base_r < 0.5 else 60)
                p.setPen(QPen(c, 2.0))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(QPointF(cx, cy), r, r)
            # Center dot
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(motion_color))
            p.drawEllipse(QPointF(cx, cy), inner_r * 0.15, inner_r * 0.15)

        elif self._motion == "Walking":
            # Three animated chevron arrows pointing upward, scrolling in phase
            p.setPen(QPen(motion_color, 2.5, Qt.PenStyle.SolidLine,
                          Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            arrow_w = inner_r * 0.45
            arrow_h = inner_r * 0.25
            for i in range(3):
                # Each chevron scrolls upward, wrapping with phase
                offset = ((self._phase * inner_r * 0.5 / math.pi) + i * inner_r * 0.55) % (inner_r * 1.4)
                y_base = cy + inner_r * 0.55 - offset
                if abs(y_base - cy) > inner_r * 0.85:
                    continue
                alpha = max(40, 255 - int(abs(y_base - cy) / inner_r * 280))
                c = QColor(motion_color)
                c.setAlpha(alpha)
                p.setPen(QPen(c, 2.5, Qt.PenStyle.SolidLine,
                              Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
                p.drawLine(QPointF(cx - arrow_w, y_base + arrow_h),
                           QPointF(cx,            y_base))
                p.drawLine(QPointF(cx,            y_base),
                           QPointF(cx + arrow_w, y_base + arrow_h))
            # Center dot
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(motion_color))
            p.drawEllipse(QPointF(cx, cy), inner_r * 0.15, inner_r * 0.15)

        elif self._motion == "Moving":
            # Radial burst lines animated outward
            p.setPen(QPen(motion_color, 2.5, Qt.PenStyle.SolidLine,
                          Qt.PenCapStyle.RoundCap))
            for i in range(6):
                angle = self._phase + i * (math.pi / 3)
                r_in  = inner_r * 0.20
                r_out = inner_r * (0.55 + 0.22 * math.sin(self._phase * 4 + i))
                p.drawLine(QPointF(cx + math.cos(angle) * r_in,
                                   cy + math.sin(angle) * r_in),
                           QPointF(cx + math.cos(angle) * r_out,
                                   cy + math.sin(angle) * r_out))
            # Center dot
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(motion_color))
            p.drawEllipse(QPointF(cx, cy), inner_r * 0.18, inner_r * 0.18)

        else:
            p.setPen(QPen(QColor(config.gui_theme.subtext), 1.0))
            p.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, "–")

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
            QLabel {{ background-color: transparent; }}
        """)

        root = QHBoxLayout(self)
        root.setContentsMargins(15, 15, 15, 15)
        root.setSpacing(15)

        graphic_layout = QVBoxLayout()
        self.posture_widget = PostureWidget()
        self.posture_widget.setMinimumSize(95, 95)
        self.motion_widget  = MotionWidget()
        self.motion_widget.setMinimumSize(95, 95)
        graphic_layout.addWidget(self.posture_widget)
        graphic_layout.addWidget(self.motion_widget)

        text_layout = QVBoxLayout()
        title = QLabel("🧍 Posture & Motion")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {config.gui_theme.text};")
        text_layout.addWidget(title)

        self._labels: dict = {}
        for key in ("Posture", "Posture conf.", "Height (Range)", "Motion"):
            row     = QHBoxLayout()
            lbl_key = QLabel(key)
            lbl_key.setStyleSheet(f"color: {config.gui_theme.subtext}; font-size: 20px;")
            lbl_val = QLabel("--")
            lbl_val.setStyleSheet(
                f"color: {config.gui_theme.text}; font-size: 20px; font-weight: bold;")
            lbl_val.setAlignment(Qt.AlignmentFlag.AlignRight)
            row.addWidget(lbl_key)
            row.addWidget(lbl_val)
            text_layout.addLayout(row)
            self._labels[key] = lbl_val

        root.addLayout(graphic_layout, stretch=1)
        root.addLayout(text_layout,    stretch=2)

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
            QLabel {{ background-color: transparent; }}
        """)


# ─── Radar Posture Map Item ──────────────────────────────────────────
class RadarPostureItem(pg.GraphicsObject):
    """
    Renders the posture anatomy on the radar map at a fixed pixel size,
    plus an animated motion indicator:
      Resting → slow pulsing outer ring (green)
      Moving  → 3 orbiting amber dots at 120° spacing
    """
    def __init__(self):
        super().__init__()
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self._posture    = "Unknown"
        self._confidence = 0.0
        self._motion     = "Unknown"   # "Resting" | "Moving" | "Unknown"
        self._phase      = 0.0
        self._icon_size  = 60.0        # px — increased from 40
        self.setZValue(100)

        self._timer = QTimer()
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def set_state(self, posture: str, confidence: float, motion: str = ""):
        cls, _ = _classify_motion(motion)
        if (posture != self._posture or confidence != self._confidence
                or cls != self._motion):
            self._posture    = posture
            self._confidence = confidence
            self._motion     = cls
            self.update()

    def _tick(self):
        self._phase = (self._phase + 0.06) % (2 * math.pi)
        self.update()

    def boundingRect(self):
        # Extra space for the motion indicator (orbit dots reach ~1.45× half-icon)
        half = self._icon_size * 0.85
        return QRectF(-half, -half, half * 2, half * 2)

    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        half      = self._icon_size / 2.0
        draw_rect = QRectF(-half, -half, self._icon_size, self._icon_size)

        # Dark circle background for contrast
        painter.setPen(Qt.PenStyle.NoPen)
        bg = QColor(config.gui_theme.fig_bg)
        bg.setAlpha(190)
        painter.setBrush(QBrush(bg))
        painter.drawEllipse(QPointF(0, 0), half, half)

        # Motion indicator (drawn behind the posture figure)
        if self._motion == "Resting":
            pulse    = 0.5 + 0.5 * math.sin(self._phase)
            ring_r   = half * (1.18 + 0.14 * pulse)
            ring_col = QColor("#22C55E")
            ring_col.setAlpha(int(70 + 70 * pulse))
            painter.setPen(QPen(ring_col, 2.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(0, 0), ring_r, ring_r)

        elif self._motion == "Walking":
            # Teal sweeping arc that rotates — suggests directional movement
            teal = QColor("#14B8A6")
            arc_rect = QRectF(-half * 1.35, -half * 1.35, half * 2.7, half * 2.7)
            for i in range(2):
                c = QColor(teal)
                c.setAlpha(160 - i * 70)
                painter.setPen(QPen(c, 2.5 - i * 0.8))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                start_angle = int((self._phase * 180 / math.pi + i * 90) % 360) * 16
                painter.drawArc(arc_rect, start_angle, 100 * 16)

        elif self._motion == "Moving":
            dot_orbit = half * 1.40
            dot_size  = half * 0.20
            painter.setPen(Qt.PenStyle.NoPen)
            for i in range(3):
                angle  = self._phase * 1.6 + i * (2 * math.pi / 3)
                dx     = math.cos(angle) * dot_orbit
                dy     = math.sin(angle) * dot_orbit
                fade   = int(150 + 90 * math.sin(self._phase * 2 + i * 2.1))
                dot_c  = QColor("#F59E0B")
                dot_c.setAlpha(max(60, min(255, fade)))
                painter.setBrush(QBrush(dot_c))
                painter.drawEllipse(QPointF(dx, dy), dot_size, dot_size)

        # Posture figure
        base_color = (QColor(config.gui_theme.text)
                      if self._posture != "Fallen" else QColor("#EF4444"))
        if self._confidence < 30 and self._posture != "Fallen":
            base_color = QColor(config.gui_theme.subtext)

        inset      = self._icon_size * 0.07
        figure_rect = draw_rect.adjusted(inset, inset, -inset, -inset)
        POSTURE_PAINTERS.get(self._posture, paint_unknown)(
            painter, figure_rect, base_color, self._phase)
