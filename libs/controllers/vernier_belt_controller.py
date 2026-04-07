import threading
import datetime
import logging
import time
# from libs.controllers.gdx import gdx
from config import config

logger = logging.getLogger(__name__)

class VernierBeltControllerThread(threading.Thread):
    def __init__(self, 
                 vernier_belt_connection_q=None, 
                 vernier_belt_ref_q=None, 
                 vernier_belt_realtime_q=None,
                 start_vernier_belt_q=None,
                 sensors=None, 
                 period=None, 
                 **kwargs):
        super().__init__()
        
        # Use config defaults if not provided
        self.sensors = sensors or config.vernier.sensors
        self.period = period or int(1000 / config.vernier.rate_hz) # ms
        
        self.vernier_belt_ref_q = vernier_belt_ref_q
        self.vernier_belt_realtime_q = vernier_belt_realtime_q
        self.vernier_belt_connection_q = vernier_belt_connection_q
        self.start_vernier_belt_q = start_vernier_belt_q
        
        self.start_recording_flag = False
        self.running = True
        self.gdx = None
        
        if self.vernier_belt_connection_q:
            self.vernier_belt_connection_q.put(0) 
            
        print("Vernier belt initialized ...")

    def run(self):
        try:
            self.gdx = gdx()
            if config.vernier.use_ble:
                print(f"Connecting to Vernier Belt via BLE: {config.vernier.model}...")
                self.gdx.open_ble(config.vernier.model) 
            else:
                print("Connecting to Vernier Belt via USB...")
                self.gdx.open_usb()
                
            self.gdx.select_sensors(self.sensors)
            print(f"Vernier Belt Device connected: {config.vernier.mac}")
            
            if self.vernier_belt_connection_q:
                self.vernier_belt_connection_q.put(1)
                
            self.start_recording()
        except Exception as e:
            print(f"Vernier Belt connection failed: {e}")
            if self.vernier_belt_connection_q:
                self.vernier_belt_connection_q.put(0)
            self.close_device()

    def start_recording(self):
        if not self.gdx:
            return
            
        self.gdx.start(period=self.period)
        
        # Mode control from user snippet
        mode = 0
        while mode < 2 and self.running:
            try:
                # Read data from gdx
                data = self.gdx.read()
                if data is None:
                    continue
                    
                # Support both [force, rr] or just [force]
                force = data[0] if len(data) > 0 else 0
                respiration = data[1] if len(data) > 1 else 0
                
                timeStamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                
                # Check for start/stop recording flag from queue
                if self.start_vernier_belt_q and not self.start_vernier_belt_q.empty():
                    self.start_recording_flag = self.start_vernier_belt_q.get()
                    mode += 1
                
                # Output data
                if self.start_recording_flag and self.vernier_belt_ref_q:
                    self.vernier_belt_ref_q.put([timeStamp, force, respiration]) 
                
                if self.vernier_belt_realtime_q:
                    self.vernier_belt_realtime_q.put(force)
                    
            except Exception as e:
                print(f'Vernier Respiration Belt recording issue >>> {e}')
                self.close_device()
                if self.vernier_belt_connection_q:
                    self.vernier_belt_connection_q.put(0)
                break
        
    def stop(self):
        self.running = False
        self.close_device()

    def close_device(self):
        if self.gdx:
            try:
                self.gdx.stop()
                self.gdx.close()
            except:
                pass
        print("Vernier Belt device closed.")
