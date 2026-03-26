import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from config import config
from libs.pipelines.activityPipeline import ActivityPipeline
from libs.pipelines.respirationPipeline import RespiratoryPipeline

class ProcessorThread(QThread):
    """
    Background worker thread that consumes raw radar FFT data, runs the heavy 
    mathematical pipelines, and emits the processed dictionaries to the GUI.
    """
    # Emits (activity_output_dict, respiratory_output_dict)
    data_ready = pyqtSignal(dict, dict)

    def __init__(self, pt_fft_q, parent=None):
        super().__init__(parent)
        self.pt_fft_q = pt_fft_q
        self.running = True
        
        # Instantiate the isolated math pipelines
        self.act_pipeline = ActivityPipeline(config.radar.range_idx_num, config.radar.range_resolution)
        self.resp_pipeline = RespiratoryPipeline(fps=config.radar.frame_rate)

    def run(self):
        import queue
        while self.running:
            try:
                # 1. Block and wait for at least one frame (timeout so it can exit on stop())
                fft_frame = self.pt_fft_q.get(timeout=0.1) 
                occ_out_dict = self.act_pipeline.process_frame(fft_frame)
                frames_processed = 1
                
                # 2. Rapidly drain any EXTRA frames that piled up
                while not self.pt_fft_q.empty():
                    try:
                        fft_frame = self.pt_fft_q.get_nowait()
                        occ_out_dict = self.act_pipeline.process_frame(fft_frame)
                        frames_processed += 1
                    except queue.Empty:
                        break # Queue is fully caught up!
                
                # 3. Respiratory Processing for Monitoring Zones
                resp_dict = None
                zone_name = occ_out_dict.get('zone', 'Unknown')
                
                is_monitor = False
                if zone_name in config.layout and config.layout[zone_name].get('type') == 'monitor':
                    is_monitor = True
                elif 'Bed' in zone_name:
                    is_monitor = True
                    
                if is_monitor and occ_out_dict.get('status') != "No Occupant":
                    try:
                        resp_dict = self.resp_pipeline.process(occ_out_dict, frames=frames_processed)
                    except Exception as e:
                        print(f"[Processor] Respiratory Error: {e}")
                        self.resp_pipeline._reset_state()
                else:
                    if self.resp_pipeline.frames_since_present > 0:
                        pass # print(f"[Processor] Run off-zone reset. Clearing ghost memory.")
                    self.resp_pipeline._reset_state()

                # Push results to UI thread cleanly via Qt Signal
                self.data_ready.emit(occ_out_dict, resp_dict or {})
                
            except queue.Empty:
                self.act_pipeline.empty_room()
                self.resp_pipeline._reset_state()
                self.data_ready.emit(self.act_pipeline.output_dict, {})
                continue

    def stop(self):
        self.running = False
        self.wait()
