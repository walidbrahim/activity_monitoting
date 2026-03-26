import sys
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
    print("Starting Bed Occupation Visualizer (PyQt Version)...")

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
    
    # 5. Initialize UI
    window = MainWindow()
    
    # Connect signals seamlessly across thread boundary
    processor_thread.data_ready.connect(window.update_dashboard)
    
    # Alert & Robot Logic wrapper to avoid blocking GUI
    last_state = {"status": "", "zone": ""}
    def handle_sys_events(occ_dict, resp_dict):
        status = occ_dict.get('status', "")
        zone = occ_dict.get('zone', "No Occupant Detected")
        
        if config.app.send_alert and status != last_state["status"] and "Initializing" not in status:
            if occ_dict.get('posture') == "Fallen" or 'Apnea' in status:
                print(f"State changed to: {status}. Sending alert to Watch...")
                threading.Thread(target=send_watch_alert, args=(status,), daemon=True).start()
                last_state["status"] = status
                
        # Handle Robot Zone Changes
        if robot_ctrl.enabled and zone != last_state["zone"]:
            last_state["zone"] = zone
            base_zone = zone.split(" - ")[0]
            
            target_pose = None
            if base_zone in config.layout and "arm_move" in config.layout[base_zone]:
                target_pose = config.layout[base_zone]["arm_move"]
            elif zone == "No Occupant Detected" and "Room" in config.layout and "arm_move" in config.layout["Room"]:
                target_pose = config.layout["Room"]["arm_move"]
                
            if target_pose:
                robot_ctrl.update_pose(target_pose)

    processor_thread.data_ready.connect(handle_sys_events)

    # 6. Run Application
    window.show()
    processor_thread.start()
    
    exit_code = app.exec()
    
    print("\nCleaning up processes...")
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
