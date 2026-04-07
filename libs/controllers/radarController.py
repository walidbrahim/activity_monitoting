import serial
import numpy as np
import multiprocessing
import queue
from collections import deque
import logging
import time
import struct
from config import config
import platform
import traceback

HW = config.hardware
TI = HW.ti_serial
PRE = config.preprocessing

# TODO: Make it configurable (Yaml file)
if platform.system() == 'Darwin':
    TI_CLI_SERIAL_PORT = TI.mac_cli
    SERIAL_PORT_NAME = TI.mac_data
elif platform.system() == 'Linux':
    TI_CLI_SERIAL_PORT = TI.linux_cli
    SERIAL_PORT_NAME = TI.linux_data
else:
    TI_CLI_SERIAL_PORT = TI.win_cli
    SERIAL_PORT_NAME = TI.win_data

class WisSerial:
    def __init__(self, port=SERIAL_PORT_NAME, baudrate=921600):
        self.ser = serial.Serial()
        self.ser.baudrate = baudrate
        self.ser.timeout = 2
        self.ser.port = port

    def connect(self):
        try:
            if self.ser.is_open:
                print('will close %s' % (self.ser.port,))
                self.ser.close()
            self.ser.open()
            if self.ser.is_open:
                return
        except:
            pass

        if self.ser.is_open:
            print(' Open UART success!')
            time.sleep(0.1)
            self.ser.reset_input_buffer()
        else:
            print(' Open UART fail')

    def read(self, buf_len):
        rxbuf = self.ser.read(buf_len)
        return rxbuf
    
    def read_buffer_line(self):
        rxbuf_readline = self.ser.readline()
        return rxbuf_readline

    def write(self, content):
        if isinstance(content, str):
            content = content.encode('utf-8')
        self.ser.write(content)

    def close(self):
        try:
            self.ser.close()
        except:
            pass

    def is_open(self):
        try:
            return self.ser.is_open
        except:
            return False

class RadarController(multiprocessing.Process):
    def __init__(self, state_q, calculation_status=0, **kwargs):
        super().__init__()
        self.state_q = state_q
        self.out_fft_array_q = kwargs.get('out_fft_array_q')
        self.save_fft_array_q = kwargs.get('save_fft_array_q')
        self.pt_fft_q = kwargs.get('pt_fft_q')
        self.start_radar_flag = kwargs.get('start_radar_flag')
        self.calculation_status = calculation_status
        self.range_matrix_queue = np.zeros((0, HW.range_bins, HW.antennas), dtype=complex)
        self.fft_matrix_queue = np.zeros((0, HW.range_bins, HW.antennas), dtype=complex)
        self.able_to_calculate_flag = False
        self.RangeMatrixQueueLen = PRE.warmup_frames
        self.able_put_flag = False
        self.recording_flag = False
        self.order = 0
        self.put_fft = np.zeros((HW.range_bins, HW.antennas), dtype=complex)

    def run(self):
        logging.basicConfig(level=config.app.log_level)
        ti_cli_ser = WisSerial(SERIAL_PORT_NAME, baudrate=921600)
        self.wait_for_reconnect(ti_cli_ser, True)

    @staticmethod
    def send_ti_config(is_new_config):
        if is_new_config:
            ti_cli_ser = WisSerial(TI_CLI_SERIAL_PORT, baudrate=115200)
            ti_cli_ser.connect()
            if not ti_cli_ser.is_open():
                print(f"Skip TI config: CLI port is not open ({TI_CLI_SERIAL_PORT})")
                return
            config_path = TI.config_file
            with open(config_path, 'r') as f:
                print('\nSending Configuration to radar ...')
                config_line = f.readline()
                while config_line:
                    if not ti_cli_ser.is_open():
                        print("Stop TI config send: CLI port closed during write")
                        break
                    try:
                        ti_cli_ser.write(config_line)
                    except serial.serialutil.SerialException as exc:
                        print(f"Stop TI config send: serial error: {exc}")
                        break
                    except Exception as exc:
                        print(f"Stop TI config send: unexpected write error: {exc}")
                        break
                    time.sleep(0.1)
                    feedback = ti_cli_ser.read_buffer_line()
                    time.sleep(0.1)
                    config_line = f.readline()
                print('Radar started ...\n')

    def read_data(self, ti_cli_ser):
        is_error = False
        chirp_temp = b''
        flag_header = False
        while not is_error:
            try:
                temp_buffer = ti_cli_ser.read_buffer_line()  
                if b'\x01\x04\x03\x06\x05\x08\x07' in temp_buffer and b'TIAOP\r\n' in temp_buffer:
                    chirp_temp = temp_buffer[1:]
                    if len(chirp_temp) == 163:
                        self.analyticalBuffer(chirp_temp)
                elif b'\x01\x04\x03\x06\x05\x08\x07' in temp_buffer and b'TIAOP\r\n' not in temp_buffer:
                    flag_header = True
                    chirp_temp = temp_buffer[1:]
                elif flag_header:
                    chirp_temp += temp_buffer
                    if b'TIAOP\r\n' in temp_buffer:
                        flag_header = False
                        if len(chirp_temp) == 163:
                            self.analyticalBuffer(chirp_temp)          
            except:
                is_error = True
                traceback.print_exc()
                print('traceback.format_exc():\n%s' % traceback.format_exc())

    def wait_for_reconnect(self, ti_cli_ser, is_first_connect=False):
        while not ti_cli_ser.is_open():
            self.send_ti_config(config.app.need_send_ti_config)
            ti_cli_ser.connect()
        self.read_data(ti_cli_ser)

    def analyticalBuffer(self, data):
        header_length = 8
        timeLen = 4
        step_size = 4
        magic = struct.unpack('Q', data[:header_length])
        timeStamp = struct.unpack('I', data[header_length:(header_length + timeLen)])      
        if magic[0] == TI.magic_word[self.order]:
            content_start = header_length + timeLen
            range_matrix_real = np.zeros(HW.range_bins, dtype=int)
            range_matrix_imag = np.zeros(HW.range_bins, dtype=int)
            output_idx = 0
            for rangeIdx in range(0, HW.range_bins * step_size, step_size):
                temp_real = struct.unpack('<h', data[(content_start + rangeIdx):(content_start + rangeIdx + 2)])
                temp_imag = struct.unpack('<h', data[(content_start + rangeIdx + 2):(content_start + rangeIdx + 4)])
                range_matrix_real[output_idx] = temp_real[0]
                range_matrix_imag[output_idx] = temp_imag[0]
                output_idx = output_idx + 1
            
            endtimestamp = struct.unpack('I', data[(content_start+HW.range_bins*step_size):(content_start+HW.range_bins*step_size+4)])

            range_matrix_all_ant_real = range_matrix_real.reshape(HW.range_bins, 1)
            range_matrix_all_ant_imag = range_matrix_imag.reshape(HW.range_bins, 1)
            range_fft = range_matrix_all_ant_real + 1j * range_matrix_all_ant_imag

            self.put_fft[:, self.order] = range_fft.reshape(-1)
            self.order = self.order + 1

            if self.order == HW.antennas:
                self.order = 0
                self.pt_fft_q.put(self.put_fft)
                
                # Create a fresh array for the next frame to prevent overwriting data in the queue
                self.put_fft = np.zeros((HW.range_bins, HW.antennas), dtype=complex)
        else:
            self.order = 0
