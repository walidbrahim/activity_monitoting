# 🏠 Room Activity & Occupancy Monitoring 

An application for real-time monitoring of indoor presence and activity using 60GHz mmWave radar datasets (e.g., TI IWR6843). It supports layout geofencing, background clutter noise rejection, and vital estimation for breathing validation.

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have a virtual environment set up and activated with the necessary dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy scipy matplotlib pyserial requests
```

### 2. Run the App
Launch using your virtual environment interpreter:
```bash
venv/bin/python3 room_occupancy_app.py
```

---

## 🛠️ Project Structure

*   📅 **`room_occupancy_app.py`**: Main controller and visualization pipeline including subprocess runners.
*   📂 **`docs/diagrams/`**: Contains processing pipeline Mermaid step flowcharts:
    *   `activity_step1.mmd`: Hardware calibration and background clutter subtractions.
    *   `activity_step2.mmd`: Spatial projection to 3D room coords & vital sign gating.
    *   `activity_step3.mmd`: State machine (Occupied, Apnea, Stillness, Empty).
    *   `activity_step6.mmd`: Adaptive smoothing EMA & Coordinate actigraphy index rates.
    *   `activity_step7.mmd`: Critical fall detection logic thresholds.
    *   `activity_step8_9.mmd`: Confidence weights index generation.

---

## 🍏 MacOS Specific Setup
If you are running on a Macbook:

### Finding Serial Ports
Run this snippet to identify the radar controller interface:
```bash
ls /dev/cu.* | grep -E 'usb|SLAB|modem|serial'
```

### Display Backend
The application automatically selects the `MacOSX` backend to avoid standard `Tkinter` installation aborts (`macOS 26 requires`). Setup will fall back to `TkAgg` on Linux servers perfectly without any manual adjustments.