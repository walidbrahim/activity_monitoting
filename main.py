import sys
import time
import multiprocessing
import threading
import os
from PyQt6.QtWidgets import QApplication

from config import load_profile, ConfigFactory
from libs.controllers.radarController import RadarController
from libs.controllers.robotController import RobotController
from libs.controllers.vernier_belt_controller import VernierBeltControllerThread
from apps.bed_monitor.controller import BedMonitorController
from libs.gui.main_window import MainWindow
from libs.utils import send_watch_alert

def main():
    print("Starting Room Activity Monitoring Visualizer ...")

    # 1. Start the Qt Application
    app = QApplication(sys.argv)
    
    # 2. Setup multiprocessing queues
    state_q = multiprocessing.Queue()
    pt_fft_q = multiprocessing.Queue()
    
    # 2.5 Load layered configuration and build engine config
    base_profile = "profiles/base.yaml"
    selected = os.getenv("APP_PROFILE", "base").strip()
    if selected in ("", "base", "base.yaml", base_profile):
        profile_stack = [base_profile]
    else:
        overlay = selected if selected.endswith(".yaml") else f"{selected}.yaml"
        if "/" not in overlay:
            overlay = f"profiles/{overlay}"
        profile_stack = [base_profile, overlay]

    app_cfg = load_profile(*profile_stack)
    eng_cfg = ConfigFactory.engine_config(app_cfg)
    print(f"Successfully loaded configuration stack: {profile_stack}")

    # 2.6 Setup Vernier Belt Controller (Optional)
    vernier_belt_realtime_q = multiprocessing.Queue()
    vernier_belt_connection_q = multiprocessing.Queue()
    belt_thread = None
    if app_cfg.vernier.enabled:
        belt_thread = VernierBeltControllerThread(
            vernier_belt_connection_q=vernier_belt_connection_q,
            vernier_belt_realtime_q=vernier_belt_realtime_q,
            sensors=app_cfg.vernier.sensors,
            period=int(1000/app_cfg.vernier.rate_hz),
            use_ble=app_cfg.vernier.use_ble
        )
        belt_thread.daemon = True
        belt_thread.start()

    # 3. Start Radar Process (runs truly independent of GIL)
    radar_process = RadarController(state_q=state_q, pt_fft_q=pt_fft_q)
    radar_process.start()

    # 4. Start Background Processing Thread (New BedMonitorController)
    processor_thread = BedMonitorController(
        pt_fft_q=pt_fft_q,
        vernier_belt_realtime_q=vernier_belt_realtime_q,
        vernier_belt_connection_q=vernier_belt_connection_q,
        cfg=eng_cfg,
        belt_window_sec=eng_cfg.respiration.window_sec,
        belt_rate_hz=getattr(app_cfg.vernier, 'rate_hz', 10.0),
        recording_cfg=app_cfg.recording.model_dump(),
        db_cfg=app_cfg.database.model_dump(),
    )

    # Ensure engine starts with the configured default radar pose (not just
    # whichever layout entry is iterated first).
    default_zone = getattr(app_cfg.app, "default_radar_pose", "Room")
    default_pose = app_cfg.layout.get(default_zone, {}).get("radar_pose")
    if isinstance(default_pose, dict):
        processor_thread.update_radar_pose(default_pose)
    
    # 4.5 Start Robot Controller
    robot_ctrl = RobotController()
    if robot_ctrl.enabled:
        robot_ctrl.start()
        
        # Move to default custom pose on startup if dynamic tracking is disabled
        if not getattr(app_cfg.app, "enable_robot_arm", True):
            def_zone = getattr(app_cfg.app, "default_radar_pose", "Room")
            if def_zone in app_cfg.layout and "arm_move" in app_cfg.layout[def_zone]:
                robot_ctrl.update_pose(app_cfg.layout[def_zone]["arm_move"])
    
    # 5. Initialize UI
    window = MainWindow()
    
    # Connect signals seamlessly across thread boundary
    processor_thread.data_ready.connect(window.update_dashboard)
    
    # Alert & Robot Logic wrapper to avoid blocking GUI
    last_state = {"status": "", "zone": "No Occupant Detected", "active_robot_zone": "No Occupant Detected", "last_zone_change": time.time()}
    def handle_sys_events(occ_dict, resp_dict, frames=1):
        status = occ_dict.get('status', "")
        zone = occ_dict.get('zone', "No Occupant Detected")
        
        if app_cfg.app.send_alert and status != last_state["status"] and "Initializing" not in status:
            if occ_dict.get('posture') == "Fallen" or 'Apnea' in status:
                print(f"State changed to: {status}. Sending alert to Watch...")
                threading.Thread(target=send_watch_alert, args=(status,), daemon=True).start()
                last_state["status"] = status
                
        # Handle Robot Zone Changes with universal time delay
        if robot_ctrl.enabled and getattr(app_cfg.app, 'enable_robot_arm', True):
            # Monitor instantaneous zone switches
            if zone != last_state["zone"]:
                last_state["zone"] = zone
                last_state["last_zone_change"] = time.time()
                
            # If 2 seconds have passed holding this zone, commit to moving the robot
            if last_state["active_robot_zone"] != zone and (time.time() - last_state["last_zone_change"]) > 2.0:
                last_state["active_robot_zone"] = zone
                base_zone = zone.split(" - ")[0]
                
                target_pose = None
                radar_pose = None
                if base_zone in app_cfg.layout and "arm_move" in app_cfg.layout[base_zone]:
                    target_pose = app_cfg.layout[base_zone]["arm_move"]
                    radar_pose = app_cfg.layout[base_zone].get("radar_pose")
                else: 
                    target_pose = app_cfg.layout["Room"]["arm_move"]
                    radar_pose = app_cfg.layout["Room"].get("radar_pose")
                    
                if target_pose:
                    robot_ctrl.update_pose(target_pose)
                    
                if radar_pose:
                    room_pose = app_cfg.layout.get("Room", {}).get("radar_pose", {})
                    processor_thread.update_radar_pose(radar_pose)
                    window.update_radar_fov(
                        radar_pose.get('x', room_pose.get("x", 1.22)),
                        radar_pose.get('y', room_pose.get("y", 3.27)),
                        radar_pose.get('yaw_deg', room_pose.get("yaw_deg", 180)),
                        radar_pose.get('fov_deg', room_pose.get("fov_deg", 120))
                    )

    processor_thread.data_ready.connect(handle_sys_events)

    # 6. Run Application
    window.show()
    processor_thread.start()
    
    exit_code = app.exec()
    
    print("\nCleaning up processes...")
    if robot_ctrl.enabled:
        robot_ctrl.update_pose(app_cfg.layout["Room"]["arm_move"]) # move to home position

    processor_thread.stop()
    if belt_thread:
        belt_thread.stop()
    if robot_ctrl.enabled:
        robot_ctrl.stop()
        robot_ctrl.join(timeout=2.0)
    if radar_process.is_alive():
        radar_process.terminate()
        radar_process.join()

    sys.exit(exit_code)

if __name__ == "__main__":
    main()
