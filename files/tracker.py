"""
tracker.py
==========
Single-target CSRT tracker — robust ghost-lock prevention + HSV-histogram
confidence scoring.

Root cause of "sits on random place" (ghost-lock)
-------------------------------------------------
The old confidence metric used only Laplacian variance (sharpness).
A sharpness check cannot tell the difference between "sharp background patch"
and "sharp target". The CSRT tracker silently drifts onto high-texture
background — sharpness stays high, confidence stays high — ghost-lock.

Fix: HSV colour histogram similarity.
  - At init/re-acquisition, store the HSV histogram of the target crop.
  - Every TRACKING frame compare current bbox histogram to stored reference.
  - If Bhattacharyya similarity drops below HIST_MIN → ghost detected → SEARCHING.
  - This fires before CSRT even fails, so ghost-lock never persists.

Additional robustness
---------------------
- Gallery stores (gray, hist) pairs so search matches are histogram-verified
  before CSRT is reinitialised — eliminates false re-acquisitions.
- MAX_FAIL_COUNT = 1: one CSRT+OF failure immediately triggers SEARCHING.
- Optical flow bridge runs on every CSRT failure frame for fast-motion tolerance.
- Kalman filter runs continuously; prediction seeds SEARCHING and HUD.
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Vision utilities
# ─────────────────────────────────────────────────────────────────────────────

def _hsv_hist(bgr: np.ndarray) -> np.ndarray:
    """
    Normalised HSV histogram (H×S channels only — ignores brightness).
    Resizes crop to 64×64 first so histogram is scale-invariant.
    Returns float32 array of length 30+32=62.
    """
    small = cv2.resize(bgr, (64, 64))
    hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [30], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    hist   = np.concatenate([h_hist, s_hist]).flatten().astype(np.float32)
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def _hist_sim(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Bhattacharyya similarity: 1.0 = identical colours, 0.0 = completely different.
    """
    dist = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
    return float(max(0.0, 1.0 - dist))


def _sparse_optical_flow(prev_gray, curr_gray, pts):
    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, pts, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )
    return new_pts, status


def _match_gallery(
    frame_gray: np.ndarray,
    gallery_gray: List[np.ndarray],
    scales: Tuple[float, ...],
) -> Tuple[float, Tuple[int, int], float, int]:
    """
    Template match frame_gray against every gallery entry at every scale.
    Returns (best_score, best_loc, best_scale, best_gallery_idx).
    """
    fh, fw     = frame_gray.shape[:2]
    best_score = 0.0
    best_loc   = (0, 0)
    best_scale = 1.0
    best_idx   = 0

    for gi, tmpl in enumerate(gallery_gray):
        for scale in scales:
            tw = max(int(tmpl.shape[1] * scale), 8)
            th = max(int(tmpl.shape[0] * scale), 8)
            if tw >= fw or th >= fh:
                continue
            res = cv2.matchTemplate(
                frame_gray, cv2.resize(tmpl, (tw, th)), cv2.TM_CCOEFF_NORMED
            )
            _, val, _, loc = cv2.minMaxLoc(res)
            if val > best_score:
                best_score = float(val)
                best_loc   = loc
                best_scale = scale
                best_idx   = gi

    return best_score, best_loc, best_scale, best_idx


# ─────────────────────────────────────────────────────────────────────────────
# State machine
# ─────────────────────────────────────────────────────────────────────────────

class TrackerState(Enum):
    IDLE      = auto()
    TRACKING  = auto()
    LOST      = auto()
    SEARCHING = auto()


@dataclass
class TrackResult:
    state:           TrackerState
    bbox:            Optional[Tuple[int, int, int, int]]
    center:          Optional[Tuple[int, int]]
    confidence:      float
    smoothed_center: Optional[Tuple[float, float]]


# ─────────────────────────────────────────────────────────────────────────────
# Kalman filter
# ─────────────────────────────────────────────────────────────────────────────

class KalmanTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix    = np.array(
            [[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
        self.kf.measurementMatrix   = np.array(
            [[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32)
        self._initialized           = False

    def update(self, cx, cy):
        meas = np.array([[np.float32(cx)], [np.float32(cy)]])
        if not self._initialized:
            self.kf.statePre  = np.array([[cx],[cy],[0.],[0.]], dtype=np.float32)
            self.kf.statePost = self.kf.statePre.copy()
            self._initialized = True
        self.kf.predict()
        c = self.kf.correct(meas)
        return float(c[0][0]), float(c[1][0])

    def predict(self):
        p = self.kf.predict()
        return float(p[0][0]), float(p[1][0])

    def reset(self):
        self._initialized = False


# ─────────────────────────────────────────────────────────────────────────────
# TrackerManager
# ─────────────────────────────────────────────────────────────────────────────

class TrackerManager:
    """
    Single-target CSRT tracker with histogram-based ghost-lock detection.

    Confidence model
    ----------------
    Confidence = HSV histogram similarity between current bbox and the
    reference histogram captured at last successful init/re-acquisition.
    Range [0,1]. Falls to zero instantly if CSRT drifts onto background.

    Gallery
    -------
    Up to GALLERY_SIZE (gray, hist) pairs — time-spaced appearance snapshots.
    Every SEARCHING match is histogram-verified before CSRT reinit to prevent
    false re-acquisitions.
    """

    # Failure tolerance
    MAX_FAIL_COUNT      = 1     # 1 CSRT+OF failure → SEARCHING immediately
    MIN_CONFIDENCE      = 0.22  # histogram similarity threshold
    MIN_LOW_CONF_FRAMES = 3     # consecutive low-conf frames → ghost-lock → SEARCHING

    # Histogram ghost-lock gate
    HIST_MIN            = 0.22  # similarity below this = not our target
    HIST_VERIFY_MIN     = 0.28  # min similarity to accept a re-acquisition match

    # Template matching
    SEARCH_THRESHOLD    = 0.70
    EDGE_THRESHOLD      = 0.44  # lower at frame edges (re-entry zone)
    EDGE_STRIP_PX       = 120
    SEARCH_SCALES       = (0.65, 0.80, 1.00, 1.20, 1.45)
    MIN_TEMPLATE_DIM    = 20

    # Gallery
    GALLERY_SIZE        = 6
    GALLERY_INTERVAL    = 10     # frames between gallery additions
    GALLERY_MIN_CONF    = 0.35  # min confidence to add to gallery

    # Optical flow
    MIN_FLOW_CORNERS    = 5

    def __init__(self, tracker_type="CSRT", use_kalman=True,
                 frame_width=640, frame_height=480):
        self.tracker_type = tracker_type
        self.use_kalman   = use_kalman
        self.frame_width  = frame_width
        self.frame_height = frame_height

        self._tracker:   Optional[cv2.Tracker] = None
        self._kalman                           = KalmanTracker() if use_kalman else None
        self._state                            = TrackerState.IDLE
        self._fail_count                       = 0
        self._low_conf_count                   = 0
        self._current_bbox: Optional[Tuple]   = None
        self.reference_bbox_area: Optional[float] = None

        # Reference appearance (set at init and re-acquisition)
        self._ref_hist:   Optional[np.ndarray] = None   # HSV histogram of target

        # Best template seen during this session
        self._best_gray:  Optional[np.ndarray] = None
        self._best_hist:  Optional[np.ndarray] = None
        self._best_conf:  float                = 0.0

        # Gallery: parallel lists of (gray, hist)
        self._gallery_gray:  List[np.ndarray] = []
        self._gallery_hists: List[np.ndarray] = []
        self._gallery_counter  = 0
        self._gallery_write_ptr = 1

        # Optical flow
        self._prev_gray: Optional[np.ndarray] = None
        self._flow_pts:  Optional[np.ndarray] = None

    # ── Tracker factory ────────────────────────────────────────────────────

    def _create_tracker(self):
        pmap = {
            "CSRT":  ["TrackerCSRT_create",  "legacy.TrackerCSRT_create",  "TrackerMIL_create"],
            "KCF":   ["TrackerKCF_create",   "legacy.TrackerKCF_create",   "TrackerMIL_create"],
            "MOSSE": ["TrackerMOSSE_create", "legacy.TrackerMOSSE_create", "TrackerMIL_create"],
            "MIL":   ["TrackerMIL_create"],
        }
        candidates = pmap.get(self.tracker_type.upper(), pmap["CSRT"])
        probe_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        probe_frame[20:60, 20:70] = 200
        probe_bbox  = (20, 20, 50, 40)
        for name in candidates:
            obj = cv2
            try:
                for part in name.split("."): obj = getattr(obj, part)
                tracker = obj()
            except AttributeError:
                continue
            try:
                ok = tracker.init(probe_frame.copy(), probe_bbox)
            except Exception:
                continue
            if ok is False:
                continue
            try:
                tracker.update(probe_frame.copy())
            except Exception:
                continue
            logger.info(f"[Tracker] Using {name}")
            return obj()
        raise RuntimeError("No working OpenCV tracker. pip install --upgrade opencv-contrib-python")

    # ── Helpers ────────────────────────────────────────────────────────────

    def _is_out_of_frame(self, bbox):
        x, y, w, h = bbox
        m = 10
        cx = x + w/2; cy = y + h/2
        return (cx < m or cx > self.frame_width-m or
                cy < m or cy > self.frame_height-m)

    def _roi_bgr(self, frame, bbox):
        """Crop bbox from frame. Returns None if too small."""
        x, y, w, h = bbox
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x+w), min(fh, y+h)
        if (x2-x1) < self.MIN_TEMPLATE_DIM or (y2-y1) < self.MIN_TEMPLATE_DIM:
            return None
        return frame[y1:y2, x1:x2].copy()

    def _confidence(self, frame, bbox) -> float:
        """
        Histogram similarity between current bbox and reference histogram.
        Returns 0 immediately if region is too uniform (blur/occlusion guard).
        """
        roi = self._roi_bgr(frame, bbox)
        if roi is None:
            return 0.0
        # Sharpness gate: uniform patches (sky, walls) give meaningless histograms
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < 4.0:
            return 0.0
        if self._ref_hist is None:
            return 0.5  # no reference yet — neutral
        return _hist_sim(self._ref_hist, _hsv_hist(roi))

    def _set_reference(self, bgr_roi):
        """Store HSV histogram as the authoritative target appearance."""
        self._ref_hist = _hsv_hist(bgr_roi)

    def _transition_to_searching(self, frame, bbox):
        template = self._best_gray
        if template is None:
            roi = self._roi_bgr(frame, bbox)
            if roi is not None:
                template = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if template is not None:
            self._state = TrackerState.SEARCHING
            logger.info(
                f"[Tracker] -> SEARCHING "
                f"(gallery={len(self._gallery_gray)}, "
                f"best_conf={self._best_conf:.2f})"
            )
        else:
            self._state = TrackerState.LOST
            logger.warning("[Tracker] -> LOST (no template)")
        return self._state

    # ── Gallery ────────────────────────────────────────────────────────────

    def _update_gallery(self, bgr, gray, hist):
        """
        Slot 0 = best-confidence appearance snapshot.
        Slots 1..N = round-robin time-spaced samples.
        """
        slot0_gray = self._best_gray if self._best_gray is not None else gray
        slot0_hist = self._best_hist if self._best_hist is not None else hist

        if not self._gallery_gray:
            self._gallery_gray  = [slot0_gray]
            self._gallery_hists = [slot0_hist]
        else:
            self._gallery_gray[0]  = slot0_gray
            self._gallery_hists[0] = slot0_hist

        if len(self._gallery_gray) < self.GALLERY_SIZE:
            self._gallery_gray.append(gray)
            self._gallery_hists.append(hist)
        else:
            self._gallery_gray[self._gallery_write_ptr]  = gray
            self._gallery_hists[self._gallery_write_ptr] = hist
            self._gallery_write_ptr = (
                1 + self._gallery_write_ptr % (self.GALLERY_SIZE - 1)
            )

    # ── Optical flow ───────────────────────────────────────────────────────

    def _refresh_flow_pts(self, gray, bbox):
        x, y, w, h = bbox
        fh, fw = gray.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x+w), min(fh, y+h)
        if x2 <= x1+4 or y2 <= y1+4:
            return None
        mask = np.zeros_like(gray)
        mask[y1:y2, x1:x2] = 255
        return cv2.goodFeaturesToTrack(
            gray, maxCorners=50, qualityLevel=0.10, minDistance=3, mask=mask
        )

    def _optical_flow_update(self, curr_gray, bbox):
        if (self._prev_gray is None or self._flow_pts is None
                or len(self._flow_pts) < self.MIN_FLOW_CORNERS
                or self._prev_gray.shape != curr_gray.shape):
            return None
        new_pts, status = _sparse_optical_flow(
            self._prev_gray, curr_gray, self._flow_pts
        )
        good = status.flatten() == 1
        if good.sum() < self.MIN_FLOW_CORNERS:
            return None
        old_xy = self._flow_pts.reshape(-1, 2)[good]
        new_xy = new_pts.reshape(-1, 2)[good]
        dx = float(np.median(new_xy[:,0] - old_xy[:,0]))
        dy = float(np.median(new_xy[:,1] - old_xy[:,1]))
        x, y, w, h = bbox
        self._flow_pts = new_xy.reshape(-1, 1, 2).astype(np.float32)
        return (int(x+dx), int(y+dy), w, h)

    # ── Search ─────────────────────────────────────────────────────────────

    def _search(self, frame, frame_gray):
        """
        Full-frame gallery search with histogram verification.
        Returns (score, bbox, found).
        """
        gallery = self._gallery_gray
        hists   = self._gallery_hists
        if not gallery:
            if self._best_gray is not None:
                gallery = [self._best_gray]
                hists   = [self._best_hist] if self._best_hist is not None else [None]
            else:
                return 0.0, None, False

        score, loc, scale, best_gi = _match_gallery(frame_gray, gallery, self.SEARCH_SCALES)

        fh, fw = frame_gray.shape[:2]
        ref    = gallery[best_gi]
        tw     = max(int(ref.shape[1] * scale), 1)
        th     = max(int(ref.shape[0] * scale), 1)

        match_cx  = loc[0] + tw / 2.0
        match_cy  = loc[1] + th / 2.0
        near_edge = (match_cx < self.EDGE_STRIP_PX or
                     match_cx > fw - self.EDGE_STRIP_PX or
                     match_cy < self.EDGE_STRIP_PX or
                     match_cy > fh - self.EDGE_STRIP_PX)
        threshold = self.EDGE_THRESHOLD if near_edge else self.SEARCH_THRESHOLD

        if score < threshold:
            return score, None, False

        # Histogram verification — reject false positives
        if self._ref_hist is not None:
            x1 = max(0, loc[0]); y1 = max(0, loc[1])
            x2 = min(fw, loc[0]+tw); y2 = min(fh, loc[1]+th)
            if x2 > x1 and y2 > y1:
                match_roi = frame[y1:y2, x1:x2]
                sim = _hist_sim(self._ref_hist, _hsv_hist(match_roi))
                if sim < self.HIST_VERIFY_MIN:
                    logger.debug(f"[Tracker] Match rejected by histogram: sim={sim:.2f}")
                    return score, None, False

        new_bbox = (loc[0], loc[1], tw, th)
        return score, new_bbox, True

    # ── Public API ─────────────────────────────────────────────────────────

    def initialize(self, frame, bbox):
        self.frame_height, self.frame_width = frame.shape[:2]
        bbox  = tuple(int(v) for v in bbox)
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        x, y, w, h = bbox
        x = max(0, min(x, self.frame_width-1))
        y = max(0, min(y, self.frame_height-1))
        w = max(1, min(w, self.frame_width-x))
        h = max(1, min(h, self.frame_height-y))
        bbox = (x, y, w, h)

        self._tracker = self._create_tracker()
        ok = self._tracker.init(frame, bbox)

        if ok is not False:
            self._state               = TrackerState.TRACKING
            self._fail_count          = 0
            self._low_conf_count      = 0
            self._current_bbox        = bbox
            self.reference_bbox_area  = max(w*h, 1.0)

            # Set reference histogram from initial selection
            roi = self._roi_bgr(frame, bbox)
            if roi is not None:
                self._set_reference(roi)
                self._best_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                self._best_hist = _hsv_hist(roi)
                self._best_conf = 1.0
                self._gallery_gray  = [self._best_gray]
                self._gallery_hists = [self._best_hist]
            else:
                self._ref_hist  = None
                self._best_gray = None
                self._best_hist = None
                self._best_conf = 0.0
                self._gallery_gray  = []
                self._gallery_hists = []

            self._gallery_counter   = 0
            self._gallery_write_ptr = 1
            self._prev_gray         = None
            self._flow_pts          = None

            if self._kalman:
                self._kalman.reset()
                self._kalman.update(bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2)
            logger.info(f"[Tracker] Init OK bbox={bbox}")
        else:
            logger.error("[Tracker] Init FAILED")
            self._state = TrackerState.LOST

        return ok

    def update(self, frame):
        # ── IDLE ──────────────────────────────────────────────────────────
        if self._state == TrackerState.IDLE or self._tracker is None:
            return TrackResult(TrackerState.IDLE, None, None, 0.0, None)

        # ── LOST ──────────────────────────────────────────────────────────
        if self._state == TrackerState.LOST:
            predicted = None
            if self._kalman:
                px, py = self._kalman.predict(); predicted = (px, py)
            return TrackResult(TrackerState.LOST, self._current_bbox, None, 0.0, predicted)

        # ── SEARCHING ─────────────────────────────────────────────────────
        if self._state == TrackerState.SEARCHING:
            predicted = None
            if self._kalman:
                px, py = self._kalman.predict(); predicted = (px, py)

            frame_c    = np.ascontiguousarray(frame, dtype=np.uint8)
            frame_gray = cv2.cvtColor(frame_c, cv2.COLOR_BGR2GRAY)
            score, new_bbox, found = self._search(frame_c, frame_gray)

            if found:
                logger.info(f"[Tracker] Re-acquired score={score:.2f} bbox={new_bbox}")
                self._tracker = self._create_tracker()
                ok = self._tracker.init(frame_c, new_bbox)
                if ok is not False:
                    # Refresh reference histogram from re-acquired region
                    roi = self._roi_bgr(frame_c, new_bbox)
                    if roi is not None:
                        self._set_reference(roi)

                    self._state               = TrackerState.TRACKING
                    self._fail_count          = 0
                    self._low_conf_count      = 0
                    self._best_gray           = None
                    self._best_hist           = None
                    self._best_conf           = 0.0
                    self._gallery_gray        = []
                    self._gallery_hists       = []
                    self._gallery_counter     = 0
                    self._gallery_write_ptr   = 1
                    self._prev_gray           = None
                    self._flow_pts            = None
                    self._current_bbox        = new_bbox
                    cx = new_bbox[0] + new_bbox[2] / 2.0
                    cy = new_bbox[1] + new_bbox[3] / 2.0
                    if self._kalman:
                        self._kalman.reset()
                        self._kalman.update(cx, cy)
                        predicted = (cx, cy)
                    return TrackResult(TrackerState.TRACKING, new_bbox,
                                       (int(cx), int(cy)), score, predicted)

            return TrackResult(TrackerState.SEARCHING, self._current_bbox,
                               None, 0.0, predicted)

        # ── TRACKING ──────────────────────────────────────────────────────
        frame     = np.ascontiguousarray(frame, dtype=np.uint8)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ok, bbox_raw = self._tracker.update(frame)

        if ok is not False and ok:
            bbox = tuple(int(v) for v in bbox_raw)

            if self._is_out_of_frame(bbox):
                logger.warning("[Tracker] OOF -> SEARCHING")
                self._current_bbox = bbox
                self._prev_gray    = curr_gray
                return TrackResult(
                    self._transition_to_searching(frame, bbox),
                    self._current_bbox, None, 0.0, None
                )

            conf = self._confidence(frame, bbox)

            # Ghost-lock detection (histogram-based)
            if conf < self.HIST_MIN:
                self._low_conf_count += 1
                logger.debug(
                    f"[Tracker] Low hist-conf {conf:.2f} "
                    f"({self._low_conf_count}/{self.MIN_LOW_CONF_FRAMES})"
                )
                if self._low_conf_count >= self.MIN_LOW_CONF_FRAMES:
                    logger.warning(
                        f"[Tracker] Ghost-lock detected (conf={conf:.2f}) -> SEARCHING"
                    )
                    self._current_bbox = bbox
                    self._prev_gray    = curr_gray
                    return TrackResult(
                        self._transition_to_searching(frame, bbox),
                        self._current_bbox, None, 0.0, None
                    )
            else:
                self._low_conf_count = 0

            # Good track
            self._fail_count   = 0
            self._current_bbox = bbox

            # Update best template and gallery
            roi = self._roi_bgr(frame, bbox)
            if roi is not None:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                hist = _hsv_hist(roi)

                if conf > self._best_conf:
                    self._best_gray = gray
                    self._best_hist = hist
                    self._best_conf = conf

                self._gallery_counter += 1
                if (self._gallery_counter % self.GALLERY_INTERVAL == 0
                        and conf >= self.GALLERY_MIN_CONF):
                    self._update_gallery(roi, gray, hist)

            self._flow_pts  = self._refresh_flow_pts(curr_gray, bbox)
            self._prev_gray = curr_gray

            cx = bbox[0] + bbox[2] / 2.0
            cy = bbox[1] + bbox[3] / 2.0
            smoothed = None
            if self._kalman:
                sx, sy = self._kalman.update(cx, cy); smoothed = (sx, sy)

            return TrackResult(TrackerState.TRACKING, bbox,
                               (int(cx), int(cy)), conf, smoothed)

        else:
            # CSRT failed — try optical flow bridge
            of_bbox = self._optical_flow_update(
                curr_gray, self._current_bbox or (0,0,1,1)
            )
            if of_bbox is not None and not self._is_out_of_frame(of_bbox):
                self._tracker = self._create_tracker()
                ok2 = self._tracker.init(frame, of_bbox)
                if ok2 is not False:
                    self._current_bbox = of_bbox
                    self._prev_gray    = curr_gray
                    cx = of_bbox[0] + of_bbox[2] / 2.0
                    cy = of_bbox[1] + of_bbox[3] / 2.0
                    smoothed = None
                    if self._kalman:
                        sx, sy = self._kalman.update(cx, cy); smoothed = (sx, sy)
                    logger.debug(f"[Tracker] OF bridge -> {of_bbox}")
                    return TrackResult(TrackerState.TRACKING, of_bbox,
                                       (int(cx), int(cy)), 0.15, smoothed)

            # Both failed
            self._fail_count += 1
            self._prev_gray   = curr_gray
            logger.warning(f"[Tracker] CSRT+OF fail {self._fail_count}/{self.MAX_FAIL_COUNT}")

            if self._fail_count >= self.MAX_FAIL_COUNT:
                logger.warning("[Tracker] -> SEARCHING")
                self._transition_to_searching(frame, self._current_bbox or (0,0,1,1))

            predicted = None
            if self._kalman:
                px, py = self._kalman.predict(); predicted = (px, py)
            return TrackResult(self._state, self._current_bbox, None, 0.0, predicted)

    def reset(self):
        self._tracker             = None
        self._state               = TrackerState.IDLE
        self._fail_count          = 0
        self._low_conf_count      = 0
        self._current_bbox        = None
        self._ref_hist            = None
        self._best_gray           = None
        self._best_hist           = None
        self._best_conf           = 0.0
        self._gallery_gray        = []
        self._gallery_hists       = []
        self._gallery_counter     = 0
        self._gallery_write_ptr   = 1
        self._prev_gray           = None
        self._flow_pts            = None
        if self._kalman: self._kalman.reset()
        logger.info("[Tracker] Reset")

    def estimate_distance_ratio(self, bbox):
        if not self.reference_bbox_area: return 1.0
        area = bbox[2] * bbox[3]
        return (area / self.reference_bbox_area) ** 0.5 if area > 0 else 1.0

    @property
    def state(self): return self._state
