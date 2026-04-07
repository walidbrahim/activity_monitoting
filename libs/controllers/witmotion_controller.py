import asyncio
import datetime
from threading import Thread
import queue
from bleak import BleakClient
import time

class WitMotionControllerThread(Thread):
    """
    Controller for WitMotion BLE IMU sensors.
    Supports real-time output of Acceleration, Angular Velocity, and Angle.
    """
    def __init__(self, witmotion_mac=None, witmotion_data_q=None, 
                 start_witmotion_q=None, witmotion_realtime_q=None, 
                 location="unknown", **kwargs):
        super().__init__()
        self.witmotion_mac_addr = witmotion_mac
        self.witmotion_data_q = witmotion_data_q or queue.Queue()
        self.start_witmotion_q = start_witmotion_q or queue.Queue()
        self.witmotion_realtime_q = witmotion_realtime_q or queue.Queue()
        
        self.location = location
        self.start_recording_flag = False
        self.running = True
        self.connected = False
        
        # UUID for WitMotion data notifications
        self.UUID_READ = '0000ffe4-0000-1000-8000-00805f9a34fb'
        
        self.daemon = True
        print(f"WitMotion [{self.location}] initialized for MAC: {self.witmotion_mac_addr}")

    def stop(self):
        self.running = False

    def run(self):
        # Create a new event loop for this thread to handle Bleak async calls
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.witmotion_main())
        except Exception as e:
            print(f"WitMotion [{self.location}] Main Loop Error: {e}")
        finally:
            loop.close()

    async def witmotion_main(self):
        # Initial jitter to prevent simultaneous connection attempts on the same adapter
        import random
        await asyncio.sleep(random.uniform(0.1, 2.0))
        
        while self.running:
            try:
                print(f"WitMotion [{self.location}] Attempting connection to {self.witmotion_mac_addr}...")
                async with BleakClient(self.witmotion_mac_addr, timeout=15.0) as client:
                    self.connected = client.is_connected
                    if self.connected:
                        print(f"WitMotion [{self.location}] Connected successfully. Waiting for service stability...")
                        await asyncio.sleep(1.0)    # Brief delay for adapter stability & auto-discovery
                        print(f"WitMotion [{self.location}] Starting data stream...")
                        await client.start_notify(self.UUID_READ, self.data_conv)
                        
                        # Keep-alive and command polling loop
                        while self.running and client.is_connected:
                            await asyncio.sleep(0.1)
                            # Check start/stop recording commands
                            if not self.start_witmotion_q.empty():
                                self.start_recording_flag = self.start_witmotion_q.get()
                        
                        await client.stop_notify(self.UUID_READ)
                    else:
                        print(f"WitMotion [{self.location}] Connection failed, retrying in 5s...")
                        await asyncio.sleep(5)
            except Exception as e:
                self.connected = False
                print(f"WitMotion [{self.location}] Connection error: {e}. Retrying in 5s...")
                await asyncio.sleep(5)

    def data_conv(self, sender, data):
        """
        Callback for BLE notifications. Parses WittMotion protocol.
        Data format: 0x55 0x61 [AXLh...ANGH]
        """
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Iterate over packets (each is usually 20 bytes starting with 0x55 0x61)
        for i in range(0, len(data), 20):
            if i + 19 >= len(data): break # Incomplete packet
            
            if data[i] == 0x55 and data[i+1] == 0x61:
                try:
                    # Acceleration data (g): / 32768 * 16
                    a_x = int.from_bytes(data[i+2:i+4], byteorder="little", signed=True) / 32768.0 * 16.0
                    a_y = int.from_bytes(data[i+4:i+6], byteorder="little", signed=True) / 32768.0 * 16.0
                    a_z = int.from_bytes(data[i+6:i+8], byteorder="little", signed=True) / 32768.0 * 16.0
                    
                    # Angular Velocity (deg/s): / 32768 * 2000
                    w_x = int.from_bytes(data[i+8:i+10], byteorder="little", signed=True) / 32768.0 * 2000.0
                    w_y = int.from_bytes(data[i+10:i+12], byteorder="little", signed=True) / 32768.0 * 2000.0
                    w_z = int.from_bytes(data[i+12:i+14], byteorder="little", signed=True) / 32768.0 * 2000.0
                    
                    # Angle (deg): / 32768 * 180
                    angl_x = int.from_bytes(data[i+14:i+16], byteorder="little", signed=True) / 32768.0 * 180.0
                    angl_y = int.from_bytes(data[i+16:i+18], byteorder="little", signed=True) / 32768.0 * 180.0
                    angl_z = int.from_bytes(data[i+18:i+20], byteorder="little", signed=True) / 32768.0 * 180.0
                    
                    payload = [a_x, a_y, a_z, w_x, w_y, w_z, angl_x, angl_y, angl_z]
                    
                    # Push to real-time queue for display
                    if self.witmotion_realtime_q:
                        # Use a small queue limit to prevent latency build-up if GUI lags
                        try:
                            self.witmotion_realtime_q.put_nowait(payload)
                        except queue.Full:
                            pass # skip frame if queue full

                    # Push to recording queue if active
                    if self.start_recording_flag and self.witmotion_data_q:
                        self.witmotion_data_q.put([timestamp] + payload)
                        
                except Exception as e:
                    print(f"WitMotion [{self.location}] Parse error at index {i}: {e}")
