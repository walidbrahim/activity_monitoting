import time
import queue
import threading

try:
    import xarm
    USE_ROBOT = True
except ImportError:
    USE_ROBOT = False

class RobotController(threading.Thread):
    def __init__(self):
        super().__init__()
        self.arm = None
        self.enabled = USE_ROBOT
        self.running = False
        self.input_queue = queue.Queue()
        self.current_positions = [500] * 6
        
        if not self.enabled:
            print("xArm library not found. Robot tracking disabled.")
            return

        self._init_arm()

    def _init_arm(self):
        try:
            print("Initializing xArm...")
            self.arm = xarm.Controller("USB", debug=False)
            self.running = True
            print("xArm Connected.")
        except Exception as e:
            print(f"xArm Connection Failed: {e}")
            self.arm = None
            self.enabled = False
            self.running = False

    def _move_arm(self, positions):
        if not self.arm: return
        try:
            # Command IDs 3, 4, 5, 6 (1 is removed, 2 holds radar and is fixed)
            cmd = [[i + 1, int(p)] for i, p in enumerate(positions) if (i + 1) not in [1, 2]]
            if cmd:
                self.arm.setPosition(cmd)
        except Exception as e:
            print(f"Arm move error: {e}")

    def update_pose(self, positions_array):
        if self.enabled and positions_array:
            if len(positions_array) == 6:
                if positions_array != self.current_positions:
                    self.current_positions = positions_array.copy()
                    self.input_queue.put(positions_array)

    def stop(self):
        self.running = False
        self.input_queue.put(None)

    def run(self):
        if not self.enabled or not self.arm:
            return

        print("RobotController thread started.")
        
        while self.running:
            try:
                data = self.input_queue.get(block=True, timeout=1.0)
                if data is None:
                    break
                    
                target_positions = data
                print(f"[Robot] Steering xArm to new zone pose: {target_positions}")
                self._move_arm(target_positions)
                
            except queue.Empty:
                pass
            except Exception as e:
                print(f"RobotController Error: {e}")
                
        print("RobotController thread stopped.")
