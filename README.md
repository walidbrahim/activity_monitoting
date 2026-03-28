# 🏠 Advanced Room Activity & Occupancy Monitoring 

An industrial-grade real-time monitoring application for indoor presence and activity using 60GHz mmWave radar (e.g., TI IWR6843). Features a high-performance **PyQt6/PyQtGraph** dashboard, multi-layered aliveness checks, and precise chest-focus vital signs tracking.

---

## 🚀 Key Features

*   🎯 **Precision Targeting**: Intelligent scoring prioritizes the occupant's chest/torso for optimal vital signs extraction.
*   🛡️ **3-Layer Aliveness Check**: Mitigates false positives from mechanical clutter (fans, curtains) using SNR floors, spatial stability tracking, and multi-metric spectral analysis (Prominence, Autocorr, Entropy).
*   📊 **Premium Dashboard**: Real-time visualization of breathing signals, respiration rates, radar maps, and occupancy trends with a modern dark theme and glassmorphism UI.
*   🗺️ **Smart Geofencing**: Support for custom zones (Bed, Chair, Transit) with dynamic color-temperature feedback based on occupancy confidence.
*   📉 **Advanced Vitals**: High-accuracy respiration and heart rate monitoring (placeholder) with trend analysis.

---

## 🛠️ Quick Start

### 1. Prerequisites
Ensure you have Python 3.9+ and a virtual environment set up:
```bash
python3 -m venv venv
source venv/bin/activate
pip install pyqt6 pyqtgraph pydantic-settings pyserial pyyaml scipy numpy
```

### 2. Run the App
Launch the main dashboard:
```bash
python3 main.py
```

---

## 📂 Project Structure

*   🏗️ **`main.py`**: Application entry point and thread management.
*   🧠 **`libs/pipelines/`**: Core processing logic (Spatial Projection, Aliveness, State Machine).
*   🎨 **`libs/gui/`**: High-performance PyQt6 dashboard implementation.
*   ⚙️ **`config.py`**: Pydantic-based configuration system.
*   📄 **`profiles/`**: YAML-based application and environment profiles.
*   🔌 **`libs/controllers/`**: Hardware interfaces for Radar and xArm.

---

## 🍎 MacOS Specifics

### Serial Port Identification
Identify the radar's CLI and Data ports:
```bash
ls /dev/cu.usbserial*
```

### Performance
The application uses **PyQt6** for a native, hardware-accelerated experience on macOS. Ensure no other application is holding the serial ports before launching.

---

## 🔧 Tuning
thresholds and parameters for filters, aliveness checks, and zone boundaries can be adjusted in `profiles/app_config.yaml`. Enable `DEBUG` log level in the config to see detailed per-frame spectral quality metrics.