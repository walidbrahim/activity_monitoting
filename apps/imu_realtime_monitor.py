import sys
import queue
import time
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt6.QtCore import QTimer
import pyqtgraph as pg

# Ensure we can import from the project root
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libs.controllers.witmotion_controller import WitMotionControllerThread
from config import load_profile

class IMUMonitorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Radar Pose Calibration")
        self.resize(800, 600)
        
        # Load config to get MAC and offsets
        try:
            self.cfg = load_profile("profiles/base.yaml", "profiles/lab.yaml")
        except Exception as e:
            print(f"Error loading config, using defaults: {e}")
            self.cfg = None
            
        self.mac = getattr(self.cfg.witmotion1, 'mac', "FE:5E:58:D8:4C:64") if self.cfg else "FE:5E:58:D8:4C:64"
        self.yaw_offset = getattr(self.cfg.app, 'imu_yaw_offset', 188.0) if self.cfg else 188.0
        self.pitch_mult = getattr(self.cfg.app, 'radar_pitch_multiplier', 1.0) if self.cfg else 1.0

        self.pitch_offset = getattr(self.cfg.app, 'imu_pitch_offset', 0.0) if self.cfg else 0.0

        print(f"Loaded Settings | MAC: {self.mac} | Yaw Offset: {self.yaw_offset} | Pitch Offset: {self.pitch_offset} | Pitch Mult: {self.pitch_mult}")

        # Setup UI
        central = QWidget()
        layout = QVBoxLayout(central)
        
        self.status_lbl = QLabel("Attempting Connection...")
        self.status_lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: white;")
        layout.addWidget(self.status_lbl)
        
        self.pose_lbl = QLabel(f"Pitch: --   |   Yaw: --")
        self.pose_lbl.setStyleSheet("font-size: 24px; color: cyan;")
        layout.addWidget(self.pose_lbl)

        # Plot Widget
        self.plot_widget = pg.PlotWidget(background='#1E1B4B')
        self.plot_widget.addLegend()
        self.plot_widget.setYRange(-180, 360)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.plot_widget)

        self.curve_pitch = self.plot_widget.plot(pen=pg.mkPen('c', width=2), name="Calculated Radar Pitch (deg)")
        self.curve_yaw = self.plot_widget.plot(pen=pg.mkPen('y', width=2), name="Calculated Radar Yaw (deg)")

        self.setCentralWidget(central)
        self.setStyleSheet("background-color: #0F172A;")

        # Data queues and history
        self.data_q = queue.Queue()
        self.cmd_q = queue.Queue()
        self.max_pts = 200
        self.pitch_hist = np.zeros(self.max_pts)
        self.yaw_hist = np.zeros(self.max_pts)

        # Start IMU Thread
        self.imu_thread = WitMotionControllerThread(
            witmotion_mac=self.mac,
            witmotion_realtime_q=self.data_q,
            start_witmotion_q=self.cmd_q,
            location="calibration_tool"
        )
        self.imu_thread.start()
        self.cmd_q.put(True)

        # Polling Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(50) # 20fps ui
        
        self.last_packet_time = time.time()

    def update_data(self):
        # Determine connection status heuristically
        if time.time() - self.last_packet_time > 2.0:
            self.status_lbl.setText("Status: Connecting / Reconnecting...")
            self.status_lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: yellow;")
        else:
            self.status_lbl.setText("Status: Stream Active")
            self.status_lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #059669;")

        samples = []
        while not self.data_q.empty():
            try: samples.append(self.data_q.get_nowait())
            except queue.Empty: break

        for s in samples:
            self.last_packet_time = time.time()
            
            a_x, a_z = s[0], s[2]
            angl_x = s[6]
            
            # ── Decoupled Pitch via Accelerometer ──
            # By calculating pitch directly from the gravity vector (a_x, a_z), 
            # we completely isolate it from the Gimbal Lock cross-coupling 
            # that affects the internal Euler angles when mounted vertically.
            import math
            raw_pitch = math.degrees(math.atan2(a_z, -a_x))
            radar_pitch = (raw_pitch + self.pitch_offset) * self.pitch_mult
            
            # Yaw remains on the internal compass/gyro integration
            radar_yaw = (angl_x + self.yaw_offset) % 360
            
            self.pitch_hist = np.roll(self.pitch_hist, -1)
            self.pitch_hist[-1] = radar_pitch
            
            self.yaw_hist = np.roll(self.yaw_hist, -1)
            self.yaw_hist[-1] = radar_yaw

        if samples:
            self.curve_pitch.setData(self.pitch_hist)
            self.curve_yaw.setData(self.yaw_hist)
            self.pose_lbl.setText(f"Radar Pitch: {self.pitch_hist[-1]:.1f}°   |   Radar Yaw: {self.yaw_hist[-1]:.1f}°")

    def closeEvent(self, event):
        self.imu_thread.running = False
        self.imu_thread.join(timeout=2.0)
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IMUMonitorApp()
    window.show()
    sys.exit(app.exec())
