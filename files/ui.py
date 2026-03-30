"""
ui.py
=====
Missile/drone seeker-style HUD — fully resolution-adaptive.

All pixel constants scale from a 640×480 baseline using:
    S = sqrt(fw² + fh²) / 800        (800 = diag of 640×480)

So at 1080p S≈2.75, at 720p S≈1.84, at 480p S=1.0.
Text sizes, line thicknesses, arm lengths, margins, PiP size — all scale.
"""

import cv2
import math
import time
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, Callable

logger = logging.getLogger(__name__)

# Baseline diagonal (640×480)
_BASE_DIAG = 800.0


def _scale(fw, fh) -> float:
    """Compute UI scale factor from frame size. Clamped to [0.6, 3.5]."""
    raw = math.sqrt(fw * fw + fh * fh) / _BASE_DIAG
    return max(0.6, min(3.5, raw))


def _i(v, s) -> int:
    """Scale a pixel constant and return int."""
    return max(1, int(round(v * s)))


# ─────────────────────────────────────────────────────────────────────────────
# Palette  (BGR)
# ─────────────────────────────────────────────────────────────────────────────

class C:
    PRI    = (0,   255,  70)
    DIM    = (0,   160,  40)
    FAINT  = (0,    55,  12)
    AMBER  = (0,   200, 255)
    RED    = (30,   30, 220)
    GMID   = (110, 110, 110)
    BLACK  = (0,     0,   0)


# ─────────────────────────────────────────────────────────────────────────────
# ROI selector
# ─────────────────────────────────────────────────────────────────────────────

class SelectionState(Enum):
    IDLE     = auto()
    DRAWING  = auto()
    SELECTED = auto()


@dataclass
class ROISelector:
    on_select: Optional[Callable] = None
    _state:    SelectionState                      = field(default=SelectionState.IDLE,  init=False)
    _start_pt: Optional[Tuple[int, int]]           = field(default=None, init=False)
    _end_pt:   Optional[Tuple[int, int]]           = field(default=None, init=False)
    _bbox:     Optional[Tuple[int, int, int, int]] = field(default=None, init=False)

    def mouse_callback(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._start_pt = (x, y); self._end_pt = (x, y)
            self._state = SelectionState.DRAWING; self._bbox = None
        elif event == cv2.EVENT_MOUSEMOVE and self._state == SelectionState.DRAWING:
            self._end_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self._state == SelectionState.DRAWING:
            self._end_pt = (x, y)
            bbox = self._compute_bbox()
            if bbox and bbox[2] > 10 and bbox[3] > 10:
                self._bbox = bbox; self._state = SelectionState.SELECTED
                if self.on_select: self.on_select(bbox)
            else:
                self._state = SelectionState.IDLE
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._state = SelectionState.IDLE; self._bbox = None

    def _compute_bbox(self):
        if not self._start_pt or not self._end_pt: return None
        x1 = min(self._start_pt[0], self._end_pt[0])
        y1 = min(self._start_pt[1], self._end_pt[1])
        x2 = max(self._start_pt[0], self._end_pt[0])
        y2 = max(self._start_pt[1], self._end_pt[1])
        return (x1, y1, x2 - x1, y2 - y1)

    def consume(self):
        if self._state == SelectionState.SELECTED and self._bbox:
            b = self._bbox; self._state = SelectionState.IDLE; self._bbox = None
            return b
        return None

    def draw_preview(self, frame, s=1.0):
        if self._state != SelectionState.DRAWING: return
        if not self._start_pt or not self._end_pt: return
        x1 = min(self._start_pt[0], self._end_pt[0])
        y1 = min(self._start_pt[1], self._end_pt[1])
        x2 = max(self._start_pt[0], self._end_pt[0])
        y2 = max(self._start_pt[1], self._end_pt[1])
        cv2.rectangle(frame, (x1, y1), (x2, y2), C.AMBER, max(1, _i(1, s)))
        _brackets(frame, x1, y1, x2-x1, y2-y1, C.AMBER, _i(12, s), max(1, _i(2, s)))
        _label(frame, "SELECT TARGET", x1, y1 - _i(6, s), C.AMBER, 0.85 * s)

    @property
    def state(self): return self._state


# ─────────────────────────────────────────────────────────────────────────────
# Drawing primitives
# ─────────────────────────────────────────────────────────────────────────────

FONT = cv2.FONT_HERSHEY_PLAIN


def _label(frame, text, x, y, color, scale=0.9, thick=1):
    t = max(1, int(thick))
    cv2.putText(frame, text, (int(x)+1, int(y)+1), FONT, scale, C.BLACK, t+1, cv2.LINE_AA)
    cv2.putText(frame, text, (int(x),   int(y)),   FONT, scale, color,   t,   cv2.LINE_AA)


def _label_right(frame, text, right_x, y, color, scale=0.9, thick=1):
    t = max(1, int(thick))
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, t)
    x = int(right_x) - tw
    cv2.putText(frame, text, (x+1, int(y)+1), FONT, scale, C.BLACK, t+1, cv2.LINE_AA)
    cv2.putText(frame, text, (x,   int(y)),   FONT, scale, color,   t,   cv2.LINE_AA)


def _hline(frame, x1, x2, y, color, thick=1):
    cv2.line(frame, (int(x1), int(y)), (int(x2), int(y)), color, max(1,int(thick)), cv2.LINE_AA)


def _vline(frame, x, y1, y2, color, thick=1):
    cv2.line(frame, (int(x), int(y1)), (int(x), int(y2)), color, max(1,int(thick)), cv2.LINE_AA)


def _rect(frame, x, y, w, h, color, thick=1):
    cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, max(1,int(thick)))


def _rect_fill(frame, x, y, w, h, color):
    cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, -1)


def _brackets(frame, x, y, w, h, color, arm=14, thick=2):
    arm = max(4, int(arm)); thick = max(1, int(thick))
    pts = [
        ((x,       y+arm),  (x,     y),   (x+arm,     y)),
        ((x+w-arm, y),      (x+w,   y),   (x+w,       y+arm)),
        ((x,       y+h-arm),(x,     y+h), (x+arm,     y+h)),
        ((x+w-arm, y+h),    (x+w,   y+h), (x+w,       y+h-arm)),
    ]
    for p0, p1, p2 in pts:
        cv2.line(frame, p0, p1, color, thick, cv2.LINE_AA)
        cv2.line(frame, p1, p2, color, thick, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Corner brackets (static, screen corners)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_corner_brackets(frame, fw, fh, s):
    pad   = _i(6,  s)
    arm   = _i(44, s)
    thick = max(2, _i(3, s))
    col   = C.PRI
    cv2.line(frame, (pad,      pad+arm),  (pad,      pad),      col, thick, cv2.LINE_AA)
    cv2.line(frame, (pad,      pad),      (pad+arm,  pad),      col, thick, cv2.LINE_AA)
    cv2.line(frame, (fw-pad,   pad+arm),  (fw-pad,   pad),      col, thick, cv2.LINE_AA)
    cv2.line(frame, (fw-pad,   pad),      (fw-pad-arm,pad),     col, thick, cv2.LINE_AA)
    cv2.line(frame, (pad,      fh-pad-arm),(pad,     fh-pad),   col, thick, cv2.LINE_AA)
    cv2.line(frame, (pad,      fh-pad),   (pad+arm,  fh-pad),   col, thick, cv2.LINE_AA)
    cv2.line(frame, (fw-pad,   fh-pad-arm),(fw-pad,  fh-pad),   col, thick, cv2.LINE_AA)
    cv2.line(frame, (fw-pad,   fh-pad),   (fw-pad-arm,fh-pad),  col, thick, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Center reticle — static, matches reference image
# ─────────────────────────────────────────────────────────────────────────────

def _draw_center_reticle(frame, cx, cy, s):
    gap   = _i(10, s)
    hbar  = _i(38, s)
    vtick = _i(20, s)
    etch  = _i(4,  s)
    r_pip = max(2, _i(4, s))
    col   = C.PRI
    lt    = max(1, _i(2, s))
    th    = max(1, _i(1, s))

    _hline(frame, cx - gap - hbar, cx - gap, cy, col, lt)
    _vline(frame, cx - gap - hbar, cy - etch, cy + etch, col, th)
    _hline(frame, cx + gap, cx + gap + hbar, cy, col, lt)
    _vline(frame, cx + gap + hbar, cy - etch, cy + etch, col, th)

    cv2.circle(frame, (cx, cy), r_pip, col, th, cv2.LINE_AA)

    _vline(frame, cx, cy + gap, cy + gap + vtick, col, lt)
    _hline(frame, cx - etch, cx + etch, cy + gap + vtick, col, th)


# ─────────────────────────────────────────────────────────────────────────────
# Tracking crosshair — follows target
# ─────────────────────────────────────────────────────────────────────────────

def _draw_tracking_crosshair(frame, tx, ty, fw, fh, col, s):
    _hline(frame, 0, fw, ty, col, max(1, _i(1, s)))
    _vline(frame, tx, 0, fh, col, max(1, _i(1, s)))
    sb = _i(6, s)
    _rect(frame, tx - sb, ty - sb, sb * 2, sb * 2, col, max(1, _i(1, s)))


# ─────────────────────────────────────────────────────────────────────────────
# Target designation box
# ─────────────────────────────────────────────────────────────────────────────

def _draw_target_box(frame, bbox, color, confidence, blink, s):
    x, y, w, h = bbox
    arm  = max(_i(8, s), min(_i(18, s), w // 4, h // 4))
    lt   = max(1, _i(2, s))
    bh_  = max(2, _i(3, s))   # confidence bar height

    _brackets(frame, x, y, w, h, color, arm, lt)
    _rect(frame, x, y, w, h, C.FAINT, 1)

    bw = int(w * max(0.0, min(1.0, confidence)))
    _rect_fill(frame, x, y + h - bh_, w, bh_, (15, 15, 15))
    if bw > 0:
        _rect_fill(frame, x, y + h - bh_, bw, bh_, color)

    ts = max(0.6, 0.85 * s)
    if blink:
        _label(frame, "LOCK", x + _i(2, s), y - _i(5, s), color, ts)
    _label(frame, f"{int(confidence * 100)}%", x + w + _i(4, s), y + _i(10, s), color, ts)


# ─────────────────────────────────────────────────────────────────────────────
# Scan sweep
# ─────────────────────────────────────────────────────────────────────────────

def _draw_scan_sweep(frame, fh, fw, t, s):
    period = 2.5
    phase  = (t % period) / period
    if phase > 0.5: phase = 1.0 - phase
    y  = int(phase * 2 * fh)
    lt = max(1, _i(2, s))
    ov = frame.copy()
    _hline(ov, 0, fw, y,            C.AMBER,       lt)
    _hline(ov, 0, fw, y - _i(3,s),  (0,100,150),   1)
    _hline(ov, 0, fw, y + _i(3,s),  (0,100,150),   1)
    cv2.addWeighted(ov, 0.35, frame, 0.65, 0, frame)


# ─────────────────────────────────────────────────────────────────────────────
# Left panel — TARGET SELECTOR + TARGET TRACKING
# ─────────────────────────────────────────────────────────────────────────────

def _draw_left_panel(frame, state_str, confidence, error_x, error_y, tracker_type, s):
    ts_h  = max(0.65, 0.90 * s)   # header text scale
    ts_b  = max(0.60, 0.82 * s)   # body text scale
    lh    = max(11, _i(15, s))    # line height px
    pad   = _i(8,  s)
    x     = pad
    y     = _i(40, s)

    sc = (C.PRI   if state_str == "TRACKING"  else
          C.AMBER  if state_str == "SEARCHING" else
          C.RED    if state_str == "LOST"      else C.DIM)
    has_target = state_str != "IDLE"

    _label(frame, "TARGET SELECTOR",           x, y, C.PRI, ts_h, 1); y += lh
    _label(frame, f"Mode    : {tracker_type}", x, y, C.DIM, ts_b, 1); y += lh
    _label(frame, "Kalman  : ON",              x, y, C.DIM, ts_b, 1); y += lh
    _label(frame, "Gallery : 6",               x, y, C.DIM, ts_b, 1); y += lh
    _label(frame, "HSV-Hist: ON",              x, y, C.DIM, ts_b, 1)

    y += lh + _i(8, s)
    _label(frame, "TARGET TRACKING",           x, y, C.PRI, ts_h, 1); y += lh
    _label(frame, f"State   : {state_str}",    x, y, sc,    ts_b, 1); y += lh
    _label(frame, f"Conf    : {int(confidence*100):3d}%" if has_target else "Conf    : ---",
           x, y, C.DIM, ts_b, 1); y += lh
    _label(frame, f"Err-X   : {error_x:+.1f}px" if has_target else "Err-X   : ---",
           x, y, C.DIM, ts_b, 1); y += lh
    _label(frame, f"Err-Y   : {error_y:+.1f}px" if has_target else "Err-Y   : ---",
           x, y, C.DIM, ts_b, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Right panel — TEST SETTINGS + TRACKER DATA (right-aligned)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_right_panel(frame, fw, fps, tracker_type, source_str, distance_ratio, target_xy, s):
    ts_h  = max(0.65, 0.90 * s)
    ts_b  = max(0.60, 0.82 * s)
    lh    = max(11, _i(15, s))
    rx    = fw - _i(8, s)
    y     = _i(40, s)

    src = str(source_str)
    src = src if len(src) <= 14 else src[-14:]

    _label_right(frame, "TEST SETTINGS",             rx, y, C.PRI,   ts_h, 1); y += lh
    _label_right(frame, f"Source  : {src.upper()}",  rx, y, C.DIM,   ts_b, 1); y += lh
    _label_right(frame, f"Tracker : {tracker_type}", rx, y, C.DIM,   ts_b, 1); y += lh
    fps_col = C.PRI if fps >= 20 else C.AMBER
    _label_right(frame, f"FPS     : {fps:04.1f}",    rx, y, fps_col, ts_b, 1)

    y += lh + _i(8, s)
    _label_right(frame, "TRACKER DATA", rx, y, C.PRI, ts_h, 1); y += lh
    if target_xy:
        _label_right(frame, f"XY    : {target_xy[0]:4d}, {target_xy[1]:4d}", rx, y, C.DIM, ts_b, 1); y += lh
        _label_right(frame, f"Range : {distance_ratio:.2f}x",                rx, y, C.DIM, ts_b, 1)
    else:
        _label_right(frame, "XY    : ---", rx, y, C.DIM, ts_b, 1); y += lh
        _label_right(frame, "Range : ---", rx, y, C.DIM, ts_b, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Seeker view PiP (bottom-right)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_seeker_view(frame, fw, fh, seeker_img, s):
    pw = _i(210, s); ph = _i(158, s)
    px = fw - pw - _i(8, s)
    py = fh - ph - _i(36, s)

    if seeker_img is not None and seeker_img.size > 0:
        try:
            resized = cv2.resize(seeker_img, (pw, ph))
            frame[py:py+ph, px:px+pw] = resized
        except Exception:
            pass

    _rect(frame, px, py, pw, ph, C.PRI, max(1, _i(1, s)))
    _brackets(frame, px, py, pw, ph, C.DIM, _i(10, s), max(1, _i(1, s)))
    _label(frame, "SEEKER VIEW", px + _i(4, s), py + _i(12, s), C.PRI, max(0.65, 0.9 * s))

    mx = px + pw // 2; my = py + ph // 2
    sb = _i(5, s)
    _rect(frame, mx - sb, my - sb, sb * 2, sb * 2, C.DIM, max(1, _i(1, s)))


# ─────────────────────────────────────────────────────────────────────────────
# Bottom command bar
# ─────────────────────────────────────────────────────────────────────────────

def _draw_bottom_bar(frame, fw, fh, s):
    bh = _i(34, s); y0 = fh - bh
    ts_h = max(0.65, 0.90 * s)
    ts_b = max(0.60, 0.82 * s)
    ov = frame.copy()
    cv2.rectangle(ov, (0, y0), (fw, fh), C.BLACK, -1)
    cv2.addWeighted(ov, 0.70, frame, 0.30, 0, frame)
    _hline(frame, 0, fw, y0, C.DIM, max(1, _i(1, s)))
    _label(frame, "TRACKER COMMANDS",
           _i(8, s), y0 + _i(13, s), C.PRI, ts_h, 1)
    _label(frame, "[DRAG] SELECT    [R] RESET    [+/-] RANGE    [Q] QUIT",
           _i(8, s), y0 + _i(27, s), C.DIM, ts_b, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Top-center title + timer
# ─────────────────────────────────────────────────────────────────────────────

def _draw_title(frame, fw, elapsed, s):
    ts_t = max(0.7, 0.95 * s)
    ts_r = max(0.6, 0.82 * s)
    title = "DRONE TARGET LOCK SYSTEM"
    (tw, _), _ = cv2.getTextSize(title, FONT, ts_t, 1)
    _label(frame, title, fw // 2 - tw // 2, _i(16, s), C.PRI, ts_t, 1)

    timer = f"T+  {int(elapsed)//60:02d}:{int(elapsed)%60:02d}"
    (rw, _), _ = cv2.getTextSize(timer, FONT, ts_r, 1)
    _label(frame, timer, fw // 2 - rw // 2, _i(29, s), C.DIM, ts_r, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Sensor mode label (bottom-center)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_sensor_mode(frame, fw, fh, state_str, blink, s):
    ts = max(0.65, 0.9 * s)
    if state_str == "IDLE":
        msg = "SENSOR MODE: STANDBY"; col = C.GMID
    elif state_str == "TRACKING":
        msg = "SENSOR MODE: TRACK";   col = C.PRI
    elif state_str == "SEARCHING":
        msg = "SENSOR MODE: SEARCH" if blink else ""; col = C.AMBER
    else:
        msg = "SENSOR MODE: LOST"   if blink else ""; col = C.RED

    if msg:
        (mw, _), _ = cv2.getTextSize(msg, FONT, ts, 1)
        _label(frame, msg, fw // 2 - mw // 2, fh - _i(40, s), col, ts)


# ─────────────────────────────────────────────────────────────────────────────
# HUD overlay
# ─────────────────────────────────────────────────────────────────────────────

class HUDOverlay:
    def __init__(self, frame_width, frame_height, dead_zone_px=20.0,
                 tracker_type="CSRT", source_str="0"):
        self.fw           = frame_width
        self.fh           = frame_height
        self.dead_zone_px = int(dead_zone_px)
        self.tracker_type = tracker_type
        self.source_str   = str(source_str)
        self._fps_hist:   list = []
        self._last_t      = time.monotonic()
        self._start_t     = time.monotonic()
        self._blink       = True
        self._blink_t     = time.monotonic()

    def _fps(self):
        now = time.monotonic(); dt = now - self._last_t; self._last_t = now
        if dt > 0: self._fps_hist.append(1.0 / dt)
        if len(self._fps_hist) > 30: self._fps_hist.pop(0)
        return sum(self._fps_hist) / max(1, len(self._fps_hist))

    def _blink_tick(self):
        now = time.monotonic()
        if now - self._blink_t > 0.5:
            self._blink = not self._blink; self._blink_t = now
        return self._blink

    def draw(
        self,
        frame,
        *,
        bbox=None,
        state_str="IDLE",
        confidence=0.0,
        distance_ratio=1.0,
        error_x=0.0,
        error_y=0.0,
        command_dict=None,
        selector=None,
    ):
        fps     = self._fps()
        blink   = self._blink_tick()
        t       = time.monotonic()
        elapsed = t - self._start_t
        fh, fw  = frame.shape[:2]
        cx, cy  = fw // 2, fh // 2
        locked  = state_str == "TRACKING"

        # Compute scale factor for this frame
        s = _scale(fw, fh)

        # ── Capture target crop BEFORE any drawing ─────────────────────────
        seeker_img = None
        if bbox is not None:
            bx, by, bw, bh = bbox
            pad = max(bw, bh)
            sx1 = max(0, bx - pad); sy1 = max(0, by - pad)
            sx2 = min(fw, bx+bw+pad); sy2 = min(fh, by+bh+pad)
            if sx2 > sx1 and sy2 > sy1:
                seeker_img = frame[sy1:sy2, sx1:sx2].copy()

        # ── Scan sweep ─────────────────────────────────────────────────────
        if state_str == "SEARCHING":
            _draw_scan_sweep(frame, fh, fw, t, s)

        # ── Corner brackets ────────────────────────────────────────────────
        _draw_corner_brackets(frame, fw, fh, s)

        # ── Static center reticle ──────────────────────────────────────────
        _draw_center_reticle(frame, cx, cy, s)

        # ── Tracking crosshair + target box ───────────────────────────────
        if bbox is not None:
            bx, by, bw, bh = bbox
            tx = bx + bw // 2
            ty = by + bh // 2
            col = (C.PRI   if locked                   else
                   C.AMBER if state_str == "SEARCHING" else C.RED)
            if not blink and state_str in ("SEARCHING", "LOST"):
                col = C.FAINT
            _draw_tracking_crosshair(frame, tx, ty, fw, fh, col, s)
            _draw_target_box(frame, bbox, col, confidence, blink, s)

            if locked and (abs(error_x) + abs(error_y)) > 5:
                mag   = max(abs(error_x) + abs(error_y), 1)
                scale = min(65 / mag, 1.0)
                cv2.arrowedLine(
                    frame, (cx, cy),
                    (int(cx + error_x * scale), int(cy + error_y * scale)),
                    C.AMBER, max(1, _i(1, s)), cv2.LINE_AA, tipLength=0.25,
                )

        # ── ROI preview ────────────────────────────────────────────────────
        if selector:
            selector.draw_preview(frame, s)

        # ── Text panels ────────────────────────────────────────────────────
        target_xy = None
        if bbox is not None:
            bx, by, bw, bh = bbox
            target_xy = (bx + bw // 2, by + bh // 2)

        _draw_left_panel(frame, state_str, confidence, error_x, error_y,
                         self.tracker_type, s)
        _draw_right_panel(frame, fw, fps, self.tracker_type,
                          self.source_str, distance_ratio, target_xy, s)

        # ── Seeker PiP ─────────────────────────────────────────────────────
        _draw_seeker_view(frame, fw, fh, seeker_img, s)

        # ── Bottom bar ─────────────────────────────────────────────────────
        _draw_bottom_bar(frame, fw, fh, s)

        # ── Title + timer ──────────────────────────────────────────────────
        _draw_title(frame, fw, elapsed, s)

        # ── Sensor mode ────────────────────────────────────────────────────
        _draw_sensor_mode(frame, fw, fh, state_str, blink, s)

        return frame
