# Drone Target Lock System — Full Technical Documentation

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [File-by-File Reference](#3-file-by-file-reference)
   - [main.py](#31-mainpy)
   - [tracker.py](#32-trackerpy)
   - [pid_controller.py](#33-pid_controllerpy)
   - [control_layer.py](#34-control_layerpy)
   - [video_input.py](#35-video_inputpy)
   - [ui.py](#36-uipy)
4. [Data Flow — Frame by Frame](#4-data-flow--frame-by-frame)
5. [Tracker Deep Dive](#5-tracker-deep-dive)
6. [PID Controller Deep Dive](#6-pid-controller-deep-dive)
7. [HUD System Deep Dive](#7-hud-system-deep-dive)
8. [Configuration Reference](#8-configuration-reference)

---

## 1. System Overview

The Drone Target Lock System is a real-time computer vision pipeline that:

1. Reads a live video stream (webcam, RTSP, UDP, or file)
2. Lets the operator draw a bounding box around any target
3. Locks onto that target using OpenCV's CSRT tracker
4. Detects and recovers from tracking failures automatically
5. Computes 4-axis flight correction commands (yaw, pitch, throttle, roll) using PID controllers
6. Sends those commands to a drone flight controller (simulated, MAVLink, or Betaflight)
7. Overlays a military-style HUD on the video feed

The system is designed to be **defence-grade reliable** — it cannot get permanently stuck on the wrong object. If the tracker drifts onto the background (ghost-lock), or loses the target entirely, it transitions into an active search mode and re-acquires using a stored gallery of target appearances.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          main.py                                │
│               DroneTargetLockSystem (main loop)                 │
└────┬──────────┬──────────┬──────────┬──────────┬───────────────┘
     │          │          │          │          │
     ▼          ▼          ▼          ▼          ▼
VideoSource  Tracker   DroneAxis  DroneCtrl   HUDOverlay
(video_     Manager   Controller  (control_   + ROISelector
 input.py)  (tracker  (pid_       layer.py)   (ui.py)
             .py)     controller
                      .py)
```

### Component Responsibilities

| Component | File | Responsibility |
|-----------|------|----------------|
| `DroneTargetLockSystem` | `main.py` | Owns all components, runs main loop, handles keyboard |
| `VideoSource` | `video_input.py` | Opens any video source, reconnects on drop |
| `TrackerManager` | `tracker.py` | Tracks target, detects ghost-lock, re-acquires on loss |
| `DroneAxisController` | `pid_controller.py` | Converts pixel error to flight commands via PID |
| `DroneController` (ABC) | `control_layer.py` | Sends flight commands to hardware or simulator |
| `HUDOverlay` | `ui.py` | Draws all visual overlays on the frame |
| `ROISelector` | `ui.py` | Handles mouse drag to select target bounding box |

### Data Types passed between components

```
VideoSource  →  frame (numpy BGR array)
ROISelector  →  bbox  (x, y, w, h) tuple in pixels
TrackerManager → TrackResult(state, bbox, center, confidence, smoothed_center)
DroneAxisController → ControlCommand(yaw, pitch, throttle, roll)  all in [-1, +1]
DroneController  →  (sends to hardware / prints to console)
HUDOverlay  →  draws directly onto frame in-place
```

---

## 3. File-by-File Reference

---

### 3.1 `main.py`

**Purpose:** Entry point and orchestrator. Creates all components, runs the frame loop, handles user input.

#### Class: `DroneTargetLockSystem`

```python
class DroneTargetLockSystem:
    def __init__(self, args)
    def setup(self)
    def run(self)
    def teardown(self)
```

##### `__init__(args)`
Stores CLI args and initialises state variables:
- `_last_command` — last `ControlCommand` sent, used for HUD display
- `_last_distance_ratio` — last known distance ratio (bbox area vs reference area)
- `_target_distance_ratio` — desired hover distance, starts at `1.0` (same size as initial selection), adjustable with `+`/`-` keys
- `_frame_count`, `_start_time` — for session FPS statistics

##### `setup()`
Initialises all components in order:
1. `VideoSource` — opens the video stream
2. `TrackerManager` — creates the tracker (uses actual frame dimensions from VideoSource)
3. `DroneAxisController` — creates PID controllers with gains from CLI args
4. `DroneController` — connects to the flight controller backend
5. `ROISelector` + `HUDOverlay` — creates UI
6. OpenCV window + mouse callback registration

##### `run()` — Main Loop

```
while True:
    1. Read frame from VideoSource
    2. Check if ROISelector has a new selection
       → if yes: reset tracker, call tracker.initialize(frame, bbox)
    3. tracker.update(frame) → TrackResult
    4. If TRACKING:
       - compute error_x, error_y from target center vs frame center
       - compute distance_ratio from bbox area
       - axis_ctrl.compute(...) → ControlCommand
    5. If SEARCHING or LOST:
       - reset PID integrators
       - use Kalman predicted center for error display
    6. drone.send_command(command)
    7. hud.draw(frame, ...) — draw all overlays
    8. cv2.imshow(...)
    9. Handle keyboard input
```

##### Keyboard Handling

| Key | Action |
|-----|--------|
| `Q` / `Esc` | Break loop, go to teardown |
| `R` | `tracker.reset()` + `axis_ctrl.reset_all()` |
| `P` | Print current PID state to console |
| `+` / `=` | `_target_distance_ratio += 0.1` (max 3.0) |
| `-` | `_target_distance_ratio -= 0.1` (min 0.1) |

##### `teardown()`
Logs session stats (total frames, elapsed time, average FPS), disconnects drone, releases video, destroys OpenCV windows.

#### CLI Arguments (`parse_args()`)

All args have sensible defaults so `python main.py` works out of the box with a webcam.

```
--source       Video source (0=webcam, path, rtsp://, udp://)
--width        Capture width hint (default 640)
--height       Capture height hint (default 480)
--fps          Capture FPS hint (default 30)
--tracker      CSRT / KCF / MOSSE / MIL (default CSRT)
--no-kalman    Disable Kalman smoothing
--kp/ki/kd     PID gains (default 0.35 / 0.02 / 0.08)
--dead-zone    PID dead zone in pixels (default 20)
--backend      sim / mavlink / betaflight (default sim)
--mavlink-conn MAVLink connection string
--serial-port  Betaflight serial port
--serial-baud  Betaflight baud rate
```

---

### 3.2 `tracker.py`

**Purpose:** The most complex module. Manages the entire target tracking lifecycle including ghost-lock detection, optical flow bridging, gallery management, and full-frame re-acquisition search.

#### Module-level functions

##### `_hsv_hist(bgr) → np.ndarray`
```
Input:  BGR image crop (any size)
Output: float32 array of 62 values (30 H bins + 32 S bins), normalised [0,1]

Steps:
  1. Resize crop to 64×64 (scale-invariant)
  2. Convert BGR → HSV
  3. Compute histogram on H channel: 30 bins, range [0, 180]
  4. Compute histogram on S channel: 32 bins, range [0, 256]
  5. Concatenate, flatten, L∞-normalise with cv2.NORM_MINMAX
```

**Why H+S only?** Hue encodes colour (red, green, blue...), Saturation encodes colour purity. Value (brightness) is deliberately excluded because it changes with lighting, shadows, and exposure — it would make the same target look different under different conditions.

**Why resize to 64×64?** So a target that is 20px wide and one that is 200px wide produce the same histogram. The tracker must work at all distances.

##### `_hist_sim(h1, h2) → float`
```
Input:  Two normalised HSV histograms
Output: Similarity score in [0.0, 1.0]
        1.0 = identical colour distribution
        0.0 = completely different colours

Method: Bhattacharyya distance (cv2.HISTCMP_BHATTACHARYYA)
        dist = Bhattacharyya(h1, h2)  → [0, 1], 0=identical
        similarity = 1.0 - dist
```

**Why Bhattacharyya?** It measures the overlap between two probability distributions. It is more robust than correlation or chi-square because it handles the case where one histogram has zero bins gracefully, and it is symmetric (distance from A to B = distance from B to A).

##### `_sparse_optical_flow(prev_gray, curr_gray, pts)`
Wrapper around `cv2.calcOpticalFlowPyrLK`. Tracks a set of corner points from one frame to the next using Lucas-Kanade pyramidal optical flow. Returns new point positions and a status array (1=tracked, 0=lost).

Parameters:
- Window size: 21×21 — large enough to handle fast motion
- Pyramid levels: 3 — handles large displacements
- Termination: 20 iterations or 0.03px convergence

##### `_match_gallery(frame_gray, gallery_gray, scales)`
```
Input:  Grayscale frame, list of grayscale template crops, scale multipliers
Output: (best_score, best_location, best_scale, best_gallery_index)

For each template in gallery:
  For each scale in scales (0.65, 0.80, 1.00, 1.20, 1.45):
    Resize template to scale × original size
    cv2.matchTemplate(frame, resized_template, TM_CCOEFF_NORMED)
    Record best score and location
Return the globally best match across all templates and scales
```

This is the **re-acquisition engine**. When the target is lost, every frame runs this full-frame search. The multi-scale approach handles the case where the target has moved closer or further away since it was last seen.

---

#### Enums and Dataclasses

##### `TrackerState`
```python
class TrackerState(Enum):
    IDLE      # No target selected
    TRACKING  # Actively tracking
    LOST      # Lost target, no template available
    SEARCHING # Lost target, searching with gallery
```

##### `TrackResult`
```python
@dataclass
class TrackResult:
    state:           TrackerState
    bbox:            Optional[Tuple[int,int,int,w,h]]  # current bounding box
    center:          Optional[Tuple[int,int]]           # raw center
    confidence:      float                              # HSV histogram similarity [0,1]
    smoothed_center: Optional[Tuple[float,float]]      # Kalman-filtered center
```

---

#### Class: `KalmanTracker`

A constant-velocity Kalman filter for target position smoothing and prediction.

**State vector:** `[x, y, vx, vy]` — position and velocity in both axes.

**Measurement vector:** `[x, y]` — only position is observed (from tracker bbox).

```
Transition matrix:      Measurement matrix:
[1 0 1 0]               [1 0 0 0]
[0 1 0 1]               [0 1 0 0]
[0 0 1 0]
[0 0 0 1]
```

**What it does:**
- `update(cx, cy)` — feed a new measurement, returns smoothed position
- `predict()` — extrapolate position forward using last known velocity (used during SEARCHING to show where target probably is)
- `reset()` — clear state, used when tracker reinitialises

The process noise (`1e-2`) and measurement noise (`1e-1`) are tuned so the filter smooths out jitter without adding too much lag.

---

#### Class: `TrackerManager`

The core tracking FSM. All tracker logic lives here.

##### Constants

```python
MAX_FAIL_COUNT      = 1     # CSRT failures before → SEARCHING
MIN_CONFIDENCE      = 0.22  # unused (kept for reference)
MIN_LOW_CONF_FRAMES = 3     # consecutive low-hist frames before ghost-lock trigger
HIST_MIN            = 0.22  # hist similarity below this = "not our target"
HIST_VERIFY_MIN     = 0.28  # min similarity to accept a re-acquisition match
SEARCH_THRESHOLD    = 0.70  # template match score needed to attempt re-acquisition
EDGE_THRESHOLD      = 0.44  # lower threshold at frame edges (target re-entering)
EDGE_STRIP_PX       = 120   # pixels from edge considered "near edge"
SEARCH_SCALES       = (0.65, 0.80, 1.00, 1.20, 1.45)  # scale factors for search
MIN_TEMPLATE_DIM    = 20    # minimum bbox dimension for reliable histogram
GALLERY_SIZE        = 6     # max number of stored appearance snapshots
GALLERY_INTERVAL    = 10    # frames between gallery additions
GALLERY_MIN_CONF    = 0.35  # min confidence to add frame to gallery
MIN_FLOW_CORNERS    = 5     # minimum optical flow points for OF to be valid
```

##### State Machine

```
        ┌──────────────────────────────────┐
        │                                  │
        ▼                                  │
      IDLE ──────────────────────────► TRACKING
      (no target)   user draws ROI      (CSRT running)
                    initialize()             │
                                             │ CSRT fails once
                                             │ OR ghost-lock detected
                                             │ (low hist conf × 3 frames)
                                             ▼
                                        SEARCHING ──────► TRACKING
                                        (gallery search)  (re-acquired)
                                             │
                                             │ no template available
                                             ▼
                                           LOST
                                        (await new ROI)
```

##### `initialize(frame, bbox)`

1. Clamps bbox to frame boundaries
2. Creates a new OpenCV tracker (`_create_tracker()`)
3. Calls `tracker.init(frame, bbox)`
4. Extracts the ROI (region of interest) from the frame
5. Calls `_set_reference(roi)` — stores HSV histogram as the authoritative target colour signature
6. Saves grayscale + hist as `_best_gray`, `_best_hist` (first gallery entry)
7. Resets Kalman filter and seeds it with initial center

##### `_create_tracker()`

Tries to create an OpenCV tracker in order of preference, with fallbacks:
```
CSRT → legacy.TrackerCSRT → MIL fallback
```
Actually probes each candidate with a dummy frame to verify it works at runtime, since the tracker API changed between OpenCV 4.x versions.

##### `_confidence(frame, bbox) → float`

```
1. Crop bbox from frame
2. Sharpness gate: if Laplacian variance < 4.0 → return 0.0
   (uniform patches like sky or walls give meaningless histograms)
3. If no reference histogram stored → return 0.5 (neutral)
4. return _hist_sim(self._ref_hist, _hsv_hist(roi))
```

The sharpness gate catches occlusion (when the bbox is covered by another object) and out-of-frame conditions where the ROI clips to a plain background.

##### `_transition_to_searching(frame, bbox)`

Determines whether to go to SEARCHING or LOST:
- SEARCHING: if `_best_gray` template is available (it always is after the first good track)
- LOST: if no template exists (should not happen normally)

##### `_update_gallery(bgr, gray, hist)`

Maintains a ring buffer of appearance snapshots:
- **Slot 0** is always the best-confidence appearance seen so far
- **Slots 1–5** are time-spaced samples (round-robin)

This ensures the gallery always contains the highest-quality template plus a spread of recent appearances, covering different poses, lighting, and partial occlusions.

##### `_refresh_flow_pts(gray, bbox)` and `_optical_flow_update(curr_gray, bbox)`

Optical flow is used as a **bridge** when CSRT fails for a single frame (fast motion, motion blur). Instead of immediately entering SEARCHING, the system:

1. Tracks the corner points (detected with `cv2.goodFeaturesToTrack`) from the previous frame to the current one
2. Computes the median displacement of all tracked points
3. Translates the bounding box by that displacement
4. Reinitialises CSRT at the new position

This handles scenarios like:
- Subject makes a sudden fast movement
- Camera shake
- Motion blur on a single frame

##### `_search(frame, frame_gray) → (score, bbox, found)`

The full re-acquisition engine:

```
1. Run _match_gallery() → best template match score and location
2. Determine threshold:
   - Near edge of frame: EDGE_THRESHOLD (0.44) — easier to match
   - Centre of frame:    SEARCH_THRESHOLD (0.70)
3. If score < threshold → return (score, None, False)
4. Histogram verification:
   - Crop the matched region from the colour frame
   - Compare HSV histogram to reference: _hist_sim(ref, match_roi)
   - If sim < HIST_VERIFY_MIN (0.28) → reject, return (score, None, False)
5. Return (score, new_bbox, True)
```

The two-stage check (template score + histogram verification) means a patch can only be accepted as the target if it is both **texturally similar** (same pattern) AND **colour similar** (same hue/saturation distribution). This almost completely eliminates false re-acquisitions.

##### `update(frame) → TrackResult`

The main per-frame function. Dispatches based on current state:

**IDLE:** Returns immediately with IDLE result.

**LOST:** Runs Kalman prediction (to show where target might be), returns LOST result.

**SEARCHING:**
- Runs Kalman prediction for HUD display
- Runs `_search()` on the current frame
- If found: reinitialises CSRT, refreshes reference histogram, resets gallery, transitions to TRACKING
- If not found: returns SEARCHING result with Kalman prediction as `smoothed_center`

**TRACKING:**
```
1. Convert frame to grayscale
2. CSRT update → (ok, bbox_raw)
3. If CSRT succeeded:
   a. Check out-of-frame → SEARCHING if true
   b. _confidence() → histogram similarity
   c. If conf < HIST_MIN for 3 consecutive frames → ghost-lock → SEARCHING
   d. Update best template if current conf > _best_conf
   e. Add to gallery every GALLERY_INTERVAL frames if conf good enough
   f. Refresh optical flow corner points
   g. Kalman update
   h. Return TRACKING result
4. If CSRT failed:
   a. Try optical flow bridge
   b. If OF succeeds → reinit CSRT at OF position, return TRACKING
   c. If OF also fails → increment _fail_count
   d. If _fail_count >= MAX_FAIL_COUNT (1) → SEARCHING
   e. Return current state result
```

---

### 3.3 `pid_controller.py`

**Purpose:** Converts pixel-space target position errors into normalized [-1, +1] flight commands.

#### `PIDConfig` dataclass

```python
@dataclass
class PIDConfig:
    kp:                      float = 0.4    # Proportional gain
    ki:                      float = 0.05   # Integral gain
    kd:                      float = 0.1    # Derivative gain
    dead_zone:               float = 20.0   # Error below this → treat as zero
    integral_limit:          float = 100.0  # Anti-windup clamp on integral
    output_limit:            float = 1.0    # Output clamp
    derivative_filter_alpha: float = 0.7    # Low-pass filter on derivative (0=raw, 1=frozen)
```

#### Class: `PIDController`

A single-axis PID controller with several practical enhancements:

##### Dead Zone
If `|error| < dead_zone`, error is treated as zero AND the integral is cleared. This prevents the drone from jittering when the target is nearly centred — it just holds position.

##### Anti-windup
The integral term is clamped to `[-integral_limit, +integral_limit]`. Without this, if the drone cannot reach the target (e.g., physical obstacle), the integral keeps accumulating until the output saturates and the drone overcorrects violently when the obstacle is removed.

##### Derivative Low-Pass Filter
```
filtered_d = alpha × prev_filtered_d + (1 - alpha) × raw_d
```
With `alpha=0.7`, the derivative term is a weighted average of 70% previous and 30% new. This prevents high-frequency noise in the pixel position (from tracker jitter) from generating huge derivative spikes that would cause the drone to shudder.

##### Variable timestep
The controller uses actual elapsed time (`dt`) between calls rather than assuming a fixed rate. If `dt > 0.5s` (e.g., system stall), it clamps to `0.033s` to prevent a huge derivative/integral step.

#### `ControlCommand` dataclass

```python
@dataclass
class ControlCommand:
    yaw:      float = 0.0   # -1=left,     +1=right
    pitch:    float = 0.0   # -1=backward, +1=forward/approach
    throttle: float = 0.0   # -1=descend,  +1=ascend
    roll:     float = 0.0   # -1=left,     +1=right
```

All values are normalized to `[-1.0, +1.0]`. The control layer converts these to hardware-specific formats (PWM µs for MAVLink/Betaflight).

#### Class: `DroneAxisController`

Manages four `PIDController` instances and maps them to drone axes.

##### Control Mapping

```
Image coordinate system:
  +X = right,  +Y = down

x_error = target_cx - frame_cx
  positive → target is to the RIGHT of center
  negative → target is to the LEFT

y_error = target_cy - frame_cy
  positive → target is BELOW center (image Y increases downward)
  negative → target is ABOVE center

dist_error = desired_distance_ratio - current_distance_ratio
  positive → target too far, need to approach
  negative → target too close, need to retreat
```

##### Axis Assignment

```
YAW      ← x_error             Rotate drone to face target horizontally
ROLL     ← x_error × 0.30      Lateral strafe (secondary, faster correction)
THROTTLE ← -y_error            Climb/descend (negated: below=descend)
PITCH    ← dist_error           Approach/retreat
           + (-y_error/frame_cy) × 0.25   Nose tilt for faster vertical response
```

The **blend coefficients** (0.30 for roll, 0.25 for vertical pitch) are tunable. Setting both to 0.0 gives pure yaw+throttle only.

**Distance ratio** is computed from bbox area:
```
current_ratio = sqrt(bbox_area / reference_bbox_area)
```
So if the target occupies twice the linear dimension as at init, `ratio = 2.0`. The PID drives this toward `desired_distance_ratio` (default 1.0, adjustable with `+`/`-`).

---

### 3.4 `control_layer.py`

**Purpose:** Abstracts flight controller communication behind a common interface. Allows swapping hardware backends without changing any other code.

#### Abstract Base Class: `DroneController`

```python
class DroneController(ABC):
    def connect(self)     → bool   # Establish connection
    def send_command(cmd) → bool   # Send ControlCommand
    def arm(self)         → bool   # Arm motors
    def disarm(self)      → bool   # Disarm motors
    def disconnect(self)           # Clean shutdown
```

Supports `with` statement (context manager) via `__enter__`/`__exit__`.

#### `SimulatedDrone`

Default backend. Prints commands to console, rate-limited to once per `log_interval` seconds (default 0.1s) to avoid flooding output at 30 FPS.

Output format:
```
[DRONE CMD #00042] YAW=+0.123  PITCH=+0.045  THROTTLE=-0.012  ROLL=+0.037
```

#### `MAVLinkDrone`

Sends `RC_CHANNELS_OVERRIDE` MAVLink messages to ArduPilot or PX4.

**Requires:** `pip install pymavlink`

RC Channel mapping (ArduPilot standard):
```
CH1 = Roll      (1000–2000 µs, center = 1500)
CH2 = Pitch
CH3 = Throttle
CH4 = Yaw
CH5–CH8 = 0 (don't override)
```

Conversion: `pwm = 1500 + value × 500`

Connection strings:
```
"udp:127.0.0.1:14550"    — SITL simulator
"/dev/ttyUSB0,57600"     — USB serial
"tcp:192.168.1.1:5760"   — TCP
```

#### `BetaflightDrone`

Sends `MSP_SET_RAW_RC` (MSP command 200) to Betaflight flight controllers via serial port.

**Requires:** `pip install pyserial`

**MSP Packet Format:**
```
$ M < [size=16] [cmd=200] [data: 8× uint16 LE] [checksum=XOR of size+cmd+data]
```

Channel order (Betaflight default): `ROLL, PITCH, THROTTLE, YAW, AUX1–4`

Note: Betaflight arming is done via AUX channel configuration in Betaflight Configurator, not via MSP command.

#### `create_controller(backend, **kwargs)`

Factory function:
```python
create_controller("sim")                              # SimulatedDrone()
create_controller("mavlink", connection_string="...") # MAVLinkDrone(...)
create_controller("betaflight", port="COM3")          # BetaflightDrone(...)
```

---

### 3.5 `video_input.py`

**Purpose:** Provides a unified `VideoSource` interface for all video inputs, with automatic reconnection on stream failure.

#### Class: `VideoSource`

##### Constructor Parameters

```python
VideoSource(
    source=0,              # int (webcam index) or str (path/URL)
    width=640,             # Desired width hint
    height=480,            # Desired height hint
    fps=30,                # Desired FPS hint
    reconnect_delay=2.0    # Seconds to wait before reconnecting
)
```

After opening, `actual_width` and `actual_height` are set from the real capture dimensions (which may differ from the hints, especially for RTSP streams).

##### `_open()`

Smart open logic:
- If source string starts with `rtsp`: uses `cv2.CAP_FFMPEG` backend and sets 5-second open/read timeouts — avoids indefinite hangs on network streams
- Otherwise: uses default `cv2.VideoCapture(source)`
- After opening: sets buffer size to 1 (`CAP_PROP_BUFFERSIZE`) to minimize latency

**Why buffer size 1?** OpenCV's default buffer holds several frames. For real-time control, you want the most recent frame, not one that is 3 frames old. Setting buffer to 1 ensures minimal latency at the cost of slightly higher CPU.

##### `read() → (bool, frame)`

If not connected, tries to reconnect before returning `(False, None)`. If `_cap.read()` fails, attempts one reconnection. On success, returns `(True, frame)`.

This means the main loop never needs to handle reconnection logic — it just checks `if not ret: continue`.

---

### 3.6 `ui.py`

**Purpose:** Draws the entire HUD overlay onto each frame. All elements scale automatically with frame resolution.

#### Resolution Scaling

All pixel constants scale relative to a **640×480 baseline**:

```python
S = sqrt(fw² + fh²) / 800.0
```

Where `800` is the diagonal of 640×480. Examples:
```
640×480  → S = 1.00  (baseline)
1280×720 → S = 1.84  (HD)
1920×1080→ S = 2.75  (Full HD)
320×240  → S = 0.60  (minimum, clamped)
```

Every pixel value is passed through `_i(base_value, s)` which returns `max(1, int(round(base_value × s)))`.

Text scales use a different formula to keep font sizes readable:
```python
text_scale = max(0.65, 0.90 * s)   # header
text_scale = max(0.60, 0.82 * s)   # body
```
The `max()` clamp ensures text is always at least minimum legible size even at very small resolutions.

#### `ROISelector`

Handles mouse events to let the user draw a bounding box:

```
EVENT_LBUTTONDOWN → store start point, state = DRAWING
EVENT_MOUSEMOVE   → update end point (live preview drawn)
EVENT_LBUTTONUP   → compute bbox, validate size > 10px, state = SELECTED
EVENT_RBUTTONDOWN → cancel, state = IDLE
```

`consume()` is called once per frame by the main loop. It returns the bbox and resets state to IDLE atomically, preventing the same selection from being used twice.

#### Drawing Primitives

| Function | Description |
|----------|-------------|
| `_label(frame, text, x, y, ...)` | Left-aligned text with black shadow |
| `_label_right(frame, text, right_x, y, ...)` | Right-aligned — measures text width, offsets x accordingly |
| `_hline` / `_vline` | Anti-aliased horizontal/vertical lines |
| `_rect` / `_rect_fill` | Rectangle outline / filled |
| `_brackets(frame, x, y, w, h, ...)` | 4 L-shaped corner brackets — used for target designation and frame corners |

#### HUD Elements

##### Corner Brackets (`_draw_corner_brackets`)
Four large L-brackets at screen corners. Always static, always visible. Arm length and line thickness scale with `s`. Give the display the characteristic mil-spec framing look.

##### Center Reticle (`_draw_center_reticle`)
Static reticle at frame center. Matches reference seeker image design:
- Short horizontal bar to the left, with vertical end-tick
- Short horizontal bar to the right, with vertical end-tick
- Small circle pip at exact center
- Short vertical tick below with horizontal end-tick

This is **not** a targeting reticle — it marks the frame center (where the drone is pointing). The target is tracked relative to this.

##### Tracking Crosshair (`_draw_tracking_crosshair`)
Full H+V lines spanning the entire frame, intersecting at the **target center** (not frame center). Also draws a small box at the intersection point. Follows the target as it moves. Color:
- Green: TRACKING
- Amber: SEARCHING
- Red: LOST

This gives instant visual feedback that the tracker is locked on — the crosshair visually "chases" the target across the frame.

##### Target Designation Box (`_draw_target_box`)
Drawn at the tracker's reported bbox:
- 4 L-brackets (corner brackets)
- Faint full-border rectangle
- Confidence bar along bottom edge (fills left-to-right with confidence %)
- Blinking "LOCK" tag above box when TRACKING
- Confidence % displayed to the right

##### Scan Sweep (`_draw_scan_sweep`)
Horizontal amber line sweeping top→bottom→top (ping-pong) during SEARCHING state. Period 2.5 seconds. Also draws faint amber halos ±3px above and below. Rendered with `addWeighted` blend (35% overlay, 65% original frame) for a ghost-scan effect.

##### Left Panel (`_draw_left_panel`)
```
TARGET SELECTOR
Mode    : CSRT
Kalman  : ON
Gallery : 6
HSV-Hist: ON

TARGET TRACKING
State   : TRACKING
Conf    :  85%
Err-X   : +12.3px
Err-Y   :  -5.1px
```

##### Right Panel (`_draw_right_panel`)
Right-aligned to 8px from right edge using `_label_right`:
```
        TEST SETTINGS
   Source  : VIDEO.MP4
   Tracker : CSRT
   FPS     : 29.8

        TRACKER DATA
   XY    :  640,  360
   Range : 1.23x
```

##### Seeker View PiP (`_draw_seeker_view`)
A picture-in-picture in the bottom-right corner showing a zoomed crop around the target. The crop is taken **before any drawing** (so the PiP shows clean video, not HUD elements). Crop region = target bbox ± 1× target dimension on each side. Resized to `210×158` (scaled). Has corner brackets and a small center box.

##### Bottom Bar (`_draw_bottom_bar`)
Semi-transparent black bar across the bottom with:
```
TRACKER COMMANDS  [DRAG] SELECT    [R] RESET    [+/-] RANGE    [Q] QUIT
```

##### Title + Timer (`_draw_title`)
Centered at the top:
```
DRONE TARGET LOCK SYSTEM
     T+  00:42
```

##### Sensor Mode (`_draw_sensor_mode`)
Centered label just above the bottom bar. Changes with state:
```
SENSOR MODE: STANDBY   (grey)
SENSOR MODE: TRACK     (green)
SENSOR MODE: SEARCH    (amber, blinking)
SENSOR MODE: LOST      (red, blinking)
```

#### `HUDOverlay.draw()` signature

```python
def draw(
    self,
    frame,
    *,
    bbox=None,              # (x,y,w,h) or None
    state_str="IDLE",       # TrackerState.name
    confidence=0.0,         # float [0,1]
    distance_ratio=1.0,     # float
    error_x=0.0,            # pixels from center
    error_y=0.0,            # pixels from center
    command_dict=None,      # ControlCommand.to_dict() (available for future use)
    selector=None,          # ROISelector instance (for drawing preview)
)
```

---

## 4. Data Flow — Frame by Frame

```
VideoSource.read()
    │
    └─► frame (BGR numpy array, e.g. 1920×1080×3 uint8)
         │
         ├─► [if new ROI] tracker.initialize(frame, bbox)
         │        Sets reference histogram, gallery, CSRT
         │
         ├─► tracker.update(frame)
         │        ├─ CSRT.update → raw bbox
         │        ├─ _confidence → HSV hist similarity
         │        ├─ ghost-lock check → SEARCHING if needed
         │        ├─ gallery update
         │        ├─ optical flow points refresh
         │        └─ Kalman.update → smoothed center
         │        Returns: TrackResult
         │
         ├─► [if TRACKING]
         │    error_x = smoothed_cx - frame_width/2
         │    error_y = smoothed_cy - frame_height/2
         │    dist_ratio = sqrt(bbox_area / ref_area)
         │    axis_ctrl.compute(tcx, tcy, dist_ratio, target_ratio)
         │         ├─ PID_yaw(error_x)     → yaw
         │         ├─ PID_roll(error_x)    → roll × 0.30
         │         ├─ PID_throttle(-error_y)→ throttle
         │         └─ PID_approach(dist_e) → pitch + blend
         │         Returns: ControlCommand
         │
         ├─► drone.send_command(ControlCommand)
         │        → SimulatedDrone: print to console
         │        → MAVLinkDrone: RC_CHANNELS_OVERRIDE MAVLink packet
         │        → BetaflightDrone: MSP_SET_RAW_RC serial packet
         │
         ├─► hud.draw(frame, bbox, state_str, confidence, ...)
         │        [modifies frame in-place]
         │        ├─ capture seeker_img from frame BEFORE drawing
         │        ├─ scan sweep (if SEARCHING)
         │        ├─ corner brackets
         │        ├─ center reticle
         │        ├─ tracking crosshair at target center
         │        ├─ target box with confidence bar
         │        ├─ error vector arrow
         │        ├─ left/right text panels
         │        ├─ seeker PiP
         │        ├─ bottom bar
         │        ├─ title + timer
         │        └─ sensor mode label
         │
         └─► cv2.imshow(frame)
```

---

## 5. Tracker Deep Dive

### Why CSRT?

CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability) is an OpenCV tracker that:
- Learns a discriminative filter on the first frame that maximises response on the target and minimises response on background
- Updates that filter online every frame
- Uses spatial reliability maps to weight reliable regions of the target more heavily

It is more accurate than KCF or MOSSE but slower (~15–30 FPS on CPU at 640×480). For higher resolutions it becomes the bottleneck, which is why the `--tracker KCF` fallback exists.

### The Ghost-Lock Problem and Solution

**Problem:** CSRT updates its filter model every frame. If the target moves out of frame or becomes occluded, CSRT doesn't know it has failed — it just continues tracking whatever it drifted onto. The bbox stays on the background, jittering around a high-texture patch. The tracker reports `ok=True` every frame. CSRT has no built-in failure detection.

**Old (broken) approach:** Laplacian variance (image sharpness) was used as confidence. The idea: if the tracked region is blurry, tracking failed. Problem: textured backgrounds (grass, gravel, walls) are just as sharp as targets. Ghost-lock survived the check.

**New approach:** HSV histogram similarity.
```
At init: store HSV histogram of target as self._ref_hist

Every TRACKING frame:
  crop = frame[bbox]
  current_hist = _hsv_hist(crop)
  conf = _hist_sim(ref_hist, current_hist)  ← Bhattacharyya similarity

If conf < 0.22 for 3 consecutive frames:
  → "This region no longer looks like our target" → SEARCHING
```

A background patch that shares the target's texture pattern will almost never share its colour distribution. This fires reliably when CSRT drifts.

### Re-acquisition Design

The gallery acts as a robust memory of what the target looks like:

```
Slot 0: Best-confidence appearance ever seen (highest HSV similarity frame)
Slot 1–5: Time-spaced samples (round-robin, updated every 10 frames)
```

Why time-spaced? The target's appearance changes as it rotates, moves, changes lighting, or becomes partially occluded. Storing only the initial appearance would fail if the target looks different when re-acquired. Storing every frame would fill the gallery with nearly-identical images. Time-spacing gives diversity.

Two-stage search verification ensures robustness:
1. Template match score must exceed threshold (texture match)
2. HSV histogram similarity must exceed `HIST_VERIFY_MIN = 0.28` (colour match)

A false positive that passes both checks is extremely rare in practice.

---

## 6. PID Controller Deep Dive

### Why 4 Axes?

A drone in free flight has 6 degrees of freedom, but for target tracking only 4 are needed:
- **Yaw**: rotate left/right to keep target horizontally centered
- **Throttle**: climb/descend to keep target vertically centered
- **Pitch**: fly forward/backward to maintain distance
- **Roll**: strafe left/right (optional blend for faster horizontal correction)

### The Blend Architecture

Pure yaw correction is slow for large angular errors — the drone has to rotate its entire body. The roll blend adds a small lateral strafe component simultaneously:

```
yaw_cmd  = PID(x_error) × 1.00   ← rotates to face target
roll_cmd = PID(x_error) × 0.30   ← strafes toward target
```

Together these produce a curved approach path that gets the target centred faster than yaw alone.

Similarly for vertical:
```
throttle = -PID(y_error) × 1.00  ← climbs/descends
pitch   +=  PID(y_error) × 0.25  ← tilts nose toward target
```

### Tuning Guide

| Parameter | Effect | Increase if | Decrease if |
|-----------|--------|-------------|-------------|
| `kp` | Response speed | Target drifts far before correcting | Drone oscillates around target |
| `ki` | Steady-state correction | Target never quite reaches center | Slow overcorrection builds up |
| `kd` | Damping | Drone oscillates past center | Drone responds too sluggishly |
| `dead_zone` | Hover stability | Drone jitters even when centered | Target can sit off-center |
| `roll_blend` | Horizontal strafe aggressiveness | Target correction is too slow | Drone moves sideways too much |

---

## 7. HUD System Deep Dive

### Resolution Adaptation

The scale factor `S` is derived from frame diagonal:

```
S = clamp(sqrt(fw² + fh²) / 800, 0.6, 3.5)
```

| Resolution | Diagonal | S |
|------------|----------|---|
| 320×240 | 400 | 0.60 (clamped) |
| 640×480 | 800 | 1.00 |
| 1280×720 | 1469 | 1.84 |
| 1920×1080 | 2203 | 2.75 |
| 3840×2160 | 4290 | 3.50 (clamped) |

Every element — margin pixels, arm lengths, text scale, line thickness, PiP size, bar height — is multiplied by S. This means the HUD occupies the same **proportional area** of the frame at any resolution.

### Text Readability

All text uses a black shadow: the text is drawn twice — once offset by (+1, +1) in black at `thick+1`, then again at the exact position in the foreground colour at `thick`. This makes text legible on any background colour without needing a backing panel.

### Seeker View PiP

The crop is taken **before any HUD drawing** to ensure the PiP shows clean video:

```python
# BEFORE drawing anything:
seeker_img = frame[sy1:sy2, sx1:sx2].copy()

# Then draw all overlays...
# Then draw PiP last, writing seeker_img into the frame
```

Crop region padding = `max(target_w, target_h)` on each side, giving context around the target. The PiP is resized to a fixed display size with `cv2.resize`.

---

## 8. Configuration Reference

### Tracker Constants (tracker.py, class TrackerManager)

| Constant | Default | Description |
|----------|---------|-------------|
| `MAX_FAIL_COUNT` | `1` | CSRT+OF failures before SEARCHING |
| `HIST_MIN` | `0.22` | Minimum hist similarity to stay in TRACKING |
| `MIN_LOW_CONF_FRAMES` | `3` | Consecutive low-conf frames before ghost-lock triggers |
| `HIST_VERIFY_MIN` | `0.28` | Minimum hist similarity to accept re-acquisition |
| `SEARCH_THRESHOLD` | `0.70` | Template match score needed for re-acquisition |
| `EDGE_THRESHOLD` | `0.44` | Template match score at frame edges |
| `EDGE_STRIP_PX` | `120` | Pixels from edge considered "near edge" |
| `SEARCH_SCALES` | `(0.65, 0.80, 1.00, 1.20, 1.45)` | Scale factors for multi-scale search |
| `GALLERY_SIZE` | `6` | Number of appearance snapshots stored |
| `GALLERY_INTERVAL` | `10` | Frames between gallery additions |
| `GALLERY_MIN_CONF` | `0.35` | Minimum confidence to add to gallery |
| `MIN_FLOW_CORNERS` | `5` | Minimum optical flow points for OF to be valid |
| `MIN_TEMPLATE_DIM` | `20` | Minimum bbox size for reliable processing |

### PID Defaults (pid_controller.py)

| Axis | kp | ki | kd | dead_zone |
|------|----|----|-----|-----------|
| YAW | 0.35 | 0.02 | 0.08 | 15.0 |
| ROLL | 0.20 | 0.01 | 0.05 | 15.0 |
| THROTTLE | 0.35 | 0.02 | 0.08 | 15.0 |
| APPROACH | 0.30 | 0.01 | 0.06 | 0.08 (ratio units) |

### HUD Palette (ui.py, class C)

| Name | BGR | Usage |
|------|-----|-------|
| `PRI` | `(0, 255, 70)` | Primary green — active/tracking elements |
| `DIM` | `(0, 160, 40)` | Dim green — secondary labels |
| `FAINT` | `(0, 55, 12)` | Near-invisible — dead zones, faint borders |
| `AMBER` | `(0, 200, 255)` | Warning — SEARCHING state |
| `RED` | `(30, 30, 220)` | Critical — LOST state |
| `GMID` | `(110, 110, 110)` | Grey — IDLE / neutral |
| `BLACK` | `(0, 0, 0)` | Text shadow |
