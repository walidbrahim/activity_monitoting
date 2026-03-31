import sys
import time
import multiprocessing
import threading
from PyQt6.QtWidgets import QApplication

from config import config
from libs.controllers.radarController import RadarController
from libs.controllers.robotController import RobotController
from libs.threads.processor_thread import ProcessorThread
from libs.gui.main_window import MainWindow
from libs.utils import send_watch_alert

def main():
    print("Starting Room Activity Monitoring Visualizer ...")

    # 1. Start the Qt Application
    app = QApplication(sys.argv)
    
    # 2. Setup multiprocessing queues
    state_q = multiprocessing.Queue()
    pt_fft_q = multiprocessing.Queue()
    
    # 3. Start Radar Process (runs truly independent of GIL)
    radar_process = RadarController(state_q=state_q, pt_fft_q=pt_fft_q)
    radar_process.start()

    # 4. Start Background Processing Thread
    processor_thread = ProcessorThread(pt_fft_q=pt_fft_q)
    
    # 4.5 Start Robot Controller
    robot_ctrl = RobotController()
    if robot_ctrl.enabled:
        robot_ctrl.start()
        
        # Move to default custom pose on startup if dynamic tracking is disabled
        if not getattr(config.app, "enable_robot_arm", True):
            def_zone = getattr(config.app, "default_radar_pose", "Room")
            if def_zone in config.layout and "arm_move" in config.layout[def_zone]:
                robot_ctrl.update_pose(config.layout[def_zone]["arm_move"])
    
    # 5. Initialize UI
    window = MainWindow()
    
    # Connect signals seamlessly across thread boundary
    processor_thread.data_ready.connect(window.update_dashboard)
    
    # Alert & Robot Logic wrapper to avoid blocking GUI
    last_state = {"status": "", "zone": "No Occupant Detected", "active_robot_zone": "No Occupant Detected", "last_zone_change": time.time()}
    def handle_sys_events(occ_dict, resp_dict):
        status = occ_dict.get('status', "")
        zone = occ_dict.get('zone', "No Occupant Detected")
        
        if config.app.send_alert and status != last_state["status"] and "Initializing" not in status:
            if occ_dict.get('posture') == "Fallen" or 'Apnea' in status:
                print(f"State changed to: {status}. Sending alert to Watch...")
                threading.Thread(target=send_watch_alert, args=(status,), daemon=True).start()
                last_state["status"] = status
                
        # Handle Robot Zone Changes with universal time delay
        if robot_ctrl.enabled and getattr(config.app, 'enable_robot_arm', True):
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
                if base_zone in config.layout and "arm_move" in config.layout[base_zone]:
                    target_pose = config.layout[base_zone]["arm_move"]
                    radar_pose = config.layout[base_zone].get("radar_pose")
                else: 
                    target_pose = config.layout["Room"]["arm_move"]
                    radar_pose = config.layout["Room"].get("radar_pose")
                    
                if target_pose:
                    robot_ctrl.update_pose(target_pose)
                    
                if radar_pose:
                    room_pose = config.layout.get("Room", {}).get("radar_pose", {})
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
        robot_ctrl.update_pose(config.layout["Room"]["arm_move"]) # move to home position

    processor_thread.stop()
    if robot_ctrl.enabled:
        robot_ctrl.stop()
        robot_ctrl.join(timeout=2.0)
    if radar_process.is_alive():
        radar_process.terminate()
        radar_process.join()

    sys.exit(exit_code)

if __name__ == "__main__":
    main()
