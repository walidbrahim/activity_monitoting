import logging
import queue
import threading
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from config import config
from libs.pipelines.activityPipeline import ActivityPipeline
from libs.pipelines.respirationPipeline import RespiratoryPipeline, RespiratoryPipelineV2

logger = logging.getLogger(__name__)

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
        self._pipeline_lock = threading.Lock()
        
        # Instantiate the isolated math pipelines
        self.act_pipeline = ActivityPipeline(config.radar.range_idx_num, config.radar.range_resolution)
        self.resp_pipeline = RespiratoryPipelineV2()

    def run(self):
        while self.running:
            try:
                # 1. Block and wait for at least one frame (timeout so it can exit on stop())
                fft_frame = self.pt_fft_q.get(timeout=0.1)

                with self._pipeline_lock:
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
                    
                    # Strip sub-zone suffix for config lookup (e.g. "Bed - Center" -> "Bed")
                    base_zone = zone_name.split(" - ")[0]
                    
                    is_monitor = False
                    if base_zone in config.layout and config.layout[base_zone].get('type') == 'monitor':
                        is_monitor = True
                        
                    if is_monitor and occ_out_dict.get('status') != "No Occupant":
                        try:
                            resp_dict = self.resp_pipeline.process(occ_out_dict, frames=frames_processed)
                        except Exception as e:
                            logger.error("Respiratory pipeline error: %s", e)
                            self.resp_pipeline._reset_state()
                    else:
                        if self.resp_pipeline.frames_since_present > 0:
                            logger.debug("Off-zone reset. Clearing respiratory ghost memory.")
                        self.resp_pipeline._reset_state()

                # Push results to UI thread cleanly via Qt Signal
                self.data_ready.emit(occ_out_dict, resp_dict or {})
                
            except queue.Empty:
                with self._pipeline_lock:
                    self.act_pipeline.empty_room()
                    self.resp_pipeline._reset_state()
                self.data_ready.emit(self.act_pipeline.output_dict, {})
                continue

    def update_radar_pose(self, pose_dict):
        """Called dynamically from the main UI thread when the xArm moves to a new zone"""
        if pose_dict:
            with self._pipeline_lock:
                self.act_pipeline.update_radar_pose(
                    x=pose_dict.get('x', self.act_pipeline.radar_x),
                    y=pose_dict.get('y', self.act_pipeline.radar_y),
                    z=pose_dict.get('z', self.act_pipeline.radar_z),
                    yaw_deg=pose_dict.get('yaw_deg', self.act_pipeline.yaw_deg),
                    pitch_deg=pose_dict.get('pitch_deg', self.act_pipeline.pitch_deg),
                    fov_deg=pose_dict.get('fov_deg', 120.0)
                )

    def stop(self):
        self.running = False
        self.wait()
