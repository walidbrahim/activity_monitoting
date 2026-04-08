"""
apps/debug/tuner_window.py
==========================
Floating parameter tuner for the Debug GUI suite.
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QDoubleSpinBox, QSpinBox, QScrollArea, QWidget, QCheckBox
)
from PyQt6.QtCore import Qt

class ParameterTunerWindow(QDialog):
    def __init__(self, parent, params: list[dict], apply_callback, palette: dict):
        super().__init__(parent)
        self.setWindowTitle("⚙️ Parameter Tuner")
        # Floating window that stays on top of the main UI
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet(f"background-color: {palette['bg']}; color: {palette['text']}; font-family: Inter;")
        self.setMinimumWidth(350)
        self.apply_callback = apply_callback
        self.params = params
        self.inputs = {}
        
        layout = QVBoxLayout(self)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        form_layout = QVBoxLayout(content)
        
        for p in self.params:
            lbl = QLabel(p["name"])
            lbl.setStyleSheet(f"color: {palette['subtext']}; font-weight: bold; font-size: 11px;")
            
            if isinstance(p["default"], bool):
                inp = QCheckBox("Enabled")
                inp.setChecked(p["default"])
                inp.setStyleSheet(f"color: {palette['text']};")
            elif isinstance(p["default"], float):
                inp = QDoubleSpinBox()
                inp.setRange(p["min"], p["max"])
                inp.setSingleStep(p.get("step", 0.01))
                if "decimals" in p: inp.setDecimals(p["decimals"])
                inp.setValue(p["default"])
            else:
                inp = QSpinBox()
                inp.setRange(int(p["min"]), int(p["max"]))
                inp.setSingleStep(int(p.get("step", 1)))
                inp.setValue(int(p["default"]))
                
            if not isinstance(inp, QCheckBox):
                inp.setStyleSheet(f"""
                    QSpinBox, QDoubleSpinBox {{
                        background: {palette['panel']}; 
                        border: 1px solid {palette['border']}; 
                        padding: 4px; 
                        border-radius: 4px;
                        color: {palette['text']};
                    }}
                """)
            
            row = QHBoxLayout()
            row.addWidget(lbl, stretch=1)
            row.addWidget(inp, stretch=1)
            form_layout.addLayout(row)
            
            self.inputs[p["id"]] = (p, inp)
            
        form_layout.addStretch()
        scroll.setWidget(content)
        scroll.setStyleSheet(f"QScrollArea {{ border: none; }}")
        
        layout.addWidget(scroll)
        
        btn_apply = QPushButton("Apply && Restart Engine")
        btn_apply.setStyleSheet(f"""
            QPushButton {{
                background: {palette['accent']}; 
                color: #000; 
                font-weight: bold; 
                padding: 10px; 
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background: {palette['cyan']};
            }}
        """)
        btn_apply.clicked.connect(self._on_apply)
        layout.addWidget(btn_apply)
        
    def _on_apply(self):
        updates = {}
        for pid, (p_def, inp) in self.inputs.items():
            if isinstance(inp, QCheckBox):
                updates[pid] = inp.isChecked()
            else:
                updates[pid] = inp.value()
        self.apply_callback(updates)

