# Drone Target Lock System

A real-time computer vision based target tracking and drone control system. Draw a bounding box on any target in the video feed and the system will lock on, track it, and output 4-axis flight control commands to keep it centered in frame.

---

## Features

- **Single-target CSRT tracking** with ghost-lock prevention via HSV histogram similarity
- **Optical flow bridge** for fast-motion recovery when CSRT fails
- **Kalman filter** for smooth position prediction
- **Full-frame gallery search** — if the target is lost, the system scans the entire frame using multi-scale template matching to re-acquire
- **HSV histogram verification** — re-acquisition matches are colour-verified before CSRT reinitialises, preventing false locks
- **4-axis PID controller** — yaw, pitch, throttle, roll with dead zone, integral windup protection, and derivative filtering
- **Three drone backends** — simulated (console), MAVLink (ArduPilot/PX4), Betaflight (MSP)
- **Military-style HUD** — seeker view PiP, tracking crosshair, left/right data panels, resolution-adaptive scaling
- **Any video source** — webcam, RTSP stream, UDP stream, or local video file

---

## Requirements

```
Python 3.9+
opencv-contrib-python
numpy
```

Install:

```bash
python -m pip install opencv-contrib-python numpy
```

Optional (for MAVLink backend):
```bash
python -m pip install pymavlink
```

Optional (for Betaflight backend):
```bash
python -m pip install pyserial
```

---

## Project Structure

```
files/
├── main.py             # Entry point — orchestrates all components
├── tracker.py          # CSRT tracker + Kalman + optical flow + gallery search
├── pid_controller.py   # 4-axis PID controller
├── control_layer.py    # Drone command dispatch (Sim / MAVLink / Betaflight)
├── video_input.py      # Unified video source (webcam / RTSP / UDP / file)
├── ui.py               # Military HUD overlay — resolution-adaptive
└── check_gpu.py        # Standalone GPU/OpenCV diagnostic tool
```

---

## Usage

### Webcam (default)
```bash
python main.py
```

### Video file
```bash
python main.py --source 'C:\path\to\video.mp4'
```

### RTSP stream
```bash
python main.py --source rtsp://192.168.1.100:554/live
```

### UDP stream
```bash
python main.py --source "udp://@:5600"
```

### MAVLink drone
```bash
python main.py --backend mavlink --mavlink-conn "udp:127.0.0.1:14550"
```

### Betaflight drone
```bash
python main.py --backend betaflight --serial-port COM3 --serial-baud 115200
```

---

## Controls

| Key | Action |
|-----|--------|
| Left-click + drag | Draw bounding box to select target |
| `R` | Reset tracker / clear target |
| `+` / `=` | Increase desired hover distance |
| `-` | Decrease desired hover distance |
| `P` | Print PID state to console |
| `Q` / `Esc` | Quit |

---

## All CLI Arguments

### Video
| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | `0` | Video source: webcam index, file path, RTSP/UDP URL |
| `--width` | `640` | Capture width hint (px) |
| `--height` | `480` | Capture height hint (px) |
| `--fps` | `30` | Capture FPS hint |

### Tracker
| Argument | Default | Description |
|----------|---------|-------------|
| `--tracker` | `CSRT` | OpenCV tracker type: `CSRT`, `KCF`, `MOSSE`, `MIL` |
| `--no-kalman` | off | Disable Kalman filter smoothing |

### PID
| Argument | Default | Description |
|----------|---------|-------------|
| `--kp` | `0.35` | Proportional gain |
| `--ki` | `0.02` | Integral gain |
| `--kd` | `0.08` | Derivative gain |
| `--dead-zone` | `20.0` | Dead zone radius in pixels |

### Drone Backend
| Argument | Default | Description |
|----------|---------|-------------|
| `--backend` | `sim` | Backend: `sim`, `mavlink`, `betaflight` |
| `--mavlink-conn` | `udp:127.0.0.1:14550` | MAVLink connection string |
| `--serial-port` | `/dev/ttyUSB0` | Serial port for Betaflight |
| `--serial-baud` | `115200` | Baud rate for Betaflight |

---

## How Tracking Works

```
User draws ROI
      │
      ▼
  initialize()
  - CSRT init
  - Store HSV histogram as reference
  - Build initial gallery entry
      │
      ▼
  update() each frame
      │
      ├─ TRACKING
      │    ├─ CSRT.update() → bbox
      │    ├─ _confidence() → HSV hist similarity vs reference
      │    │     low for 3 frames → ghost-lock detected → SEARCHING
      │    ├─ Gallery updated every 10 frames
      │    └─ Optical flow corners refreshed
      │
      ├─ SEARCHING
      │    ├─ Full-frame template match (6 gallery entries × 5 scales)
      │    ├─ Match score > threshold?
      │    │     YES → histogram verify (Bhattacharyya distance)
      │    │              pass → reinit CSRT → TRACKING
      │    │              fail → keep searching
      │    └─ Kalman prediction shown while searching
      │
      └─ LOST
           └─ No template available — awaiting new ROI selection
```

## PID Control Mapping

```
X pixel error (target left/right of center)
  → YAW      (primary)   rotate drone to face target
  → ROLL     (30% blend) lateral strafe

Y pixel error (target above/below center)
  → THROTTLE (primary)   climb / descend
  → PITCH    (25% blend) nose tilt for faster vertical response

Distance error (bbox area vs reference area)
  → PITCH    (primary)   approach / retreat
```

All outputs normalized to `[-1.0, +1.0]` and mapped to PWM `[1000, 2000 µs]` for MAVLink/Betaflight.

---

## HUD Layout

```
┌─ DRONE TARGET LOCK SYSTEM ─────────────────────────── T+ 00:00 ─┐
│ TARGET SELECTOR          [full-frame crosshair]    TEST SETTINGS │
│ Mode    : CSRT                                     Source  : 0   │
│ Kalman  : ON             [center reticle]          Tracker : CSRT│
│ Gallery : 6                                        FPS     : 30  │
│ HSV-Hist: ON                                                     │
│ TARGET TRACKING                                    TRACKER DATA  │
│ State   : TRACKING       [target brackets]         XY    :  320, 240│
│ Conf    : 85%            [tracking crosshair]      Range : 1.00x │
│ Err-X   : +12.3px                                                │
│ Err-Y   : -5.1px                  [SEEKER VIEW PiP]              │
│                    SENSOR MODE: TRACK                            │
├──────────────────────────────────────────────────────────────────┤
│ TRACKER COMMANDS  [DRAG] SELECT  [R] RESET  [+/-] RANGE  [Q] QUIT│
└──────────────────────────────────────────────────────────────────┘
```

All HUD elements scale automatically with video resolution using a diagonal-based scale factor relative to the 640×480 baseline.
