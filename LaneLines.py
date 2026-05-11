import cv2
import numpy as np
import matplotlib.image as mpimg
import time
import os

_DEBUG_LANES = os.getenv("DEBUG_LANES", "0") == "1"


def hist(img):
    bottom_half = img[img.shape[0] // 2:, :]
    return np.sum(bottom_half, axis=0)


# ============================================================
# Marking Classification Engine v4  (Connected-Component based)
# ============================================================
# Uses connected-component analysis instead of Hough line segments.
# CC is more robust because:
#   • Handles non-straight lines under perspective distortion
#   • Gap measurement is exact (blob bounding boxes, not segment endpoints)
#   • Small JPEG-artifact blobs are discarded by area filter
#   • No orientation assumption needed
# ============================================================

def clean_binary(binary):
    """3×3 close+open — preserves dash gaps (5×5 would bridge them)."""
    b = (binary > 0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k, iterations=1)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN,  k, iterations=1)
    return b


def _get_blobs(strip_u8, min_area=15, min_height=8):
    """Return list of (top, bottom) for each significant blob in strip_u8."""
    if strip_u8.max() == 0:
        return []
    _, _, stats, _ = cv2.connectedComponentsWithStats(strip_u8, connectivity=8)
    blobs = []
    for i in range(1, len(stats)):
        if stats[i, cv2.CC_STAT_AREA] >= min_area and stats[i, cv2.CC_STAT_HEIGHT] >= min_height:
            top = stats[i, cv2.CC_STAT_TOP]
            blobs.append((top, top + stats[i, cv2.CC_STAT_HEIGHT]))
    return sorted(blobs)


def _merge_blobs(blobs, tol=5):
    """Merge blobs within tol pixels of each other."""
    if not blobs:
        return []
    merged = [list(blobs[0])]
    for s, e in blobs[1:]:
        if s <= merged[-1][1] + tol:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def _classify_cc(strip, label=""):
    """
    Classify a lane strip (binary uint8) as solid / dashed / unknown
    using connected-component gap analysis.
    """
    if strip.size == 0 or strip.shape[0] < 20:
        return "unknown", 0.0, {}

    sh = strip.shape[0]
    row_active = np.any(strip > 0, axis=1)
    pixel_cov = float(np.mean(row_active))

    if pixel_cov < 0.05:
        return "unknown", pixel_cov, {}

    blobs = _get_blobs(strip)
    if not blobs:
        return ("solid", pixel_cov, {}) if pixel_cov >= 0.50 else ("unknown", pixel_cov, {})

    merged = _merge_blobs(blobs, tol=5)
    n = len(merged)
    gaps = [merged[i + 1][0] - merged[i][1] for i in range(n - 1)]
    max_gap = max(gaps, default=0)
    avg_gap = float(np.mean(gaps)) if gaps else 0.0
    covered = sum(e - s for s, e in merged)
    eff_cov = covered / sh

    if _DEBUG_LANES:
        print(f"    _classify_cc{label}: pixel_cov={pixel_cov:.2f} n_blobs={n} "
              f"max_gap={max_gap} eff_cov={eff_cov:.2f}")

    # Dashed: multiple blobs with meaningful gaps
    if n >= 2 and max_gap >= 25:
        conf = min(0.90, 0.50 + 0.05 * n + max_gap / 2000.0)
        return "dashed", conf, {"n": n, "max_gap": int(max_gap), "avg_gap": round(avg_gap, 1)}

    # Solid: one continuous high-coverage stripe
    if eff_cov >= 0.55:
        return "solid", eff_cov, {"n": n, "cov": round(eff_cov, 2)}

    # Weak dashed (far-field or compressed video dashes with smaller gaps)
    if n >= 2 and max_gap >= 12:
        return "dashed", 0.45, {"n": n, "max_gap": int(max_gap), "weak": True}

    # Low-confidence solid
    if eff_cov >= 0.25:
        return "solid", eff_cov * 0.85, {"n": n}

    return "unknown", pixel_cov, {}


def _detect_double_solid(binary, x_center, ya, search=90):
    """
    Detect double solid lines: two parallel vertical stripes side by side.
    Uses Gaussian-smoothed column-sum with flat-top suppression.

    Key fix: a single solid stripe has a flat-top column sum (all columns
    within the stripe are equally max). np.argmax returns the LEFTMOST
    maximum, so suppressing only ±10px leaves the right half of the stripe
    intact → false second peak. We now expand outward from the first peak
    to find the full flat-top extent before suppressing.
    """
    h, w = binary.shape
    xa = max(0, x_center - search)
    xb = min(w, x_center + search)
    roi = binary[ya:, xa:xb]
    if roi.shape[0] < 20:
        return None

    col = np.sum(roi > 0, axis=0).astype(np.float32)
    if len(col) >= 9:
        col = cv2.GaussianBlur(col.reshape(1, -1), (9, 1), 2.0).flatten()

    if col.max() < 3:
        return None

    idx1 = int(np.argmax(col))
    peak1 = col[idx1]

    # Expand from idx1 in both directions until col drops below 80% of peak.
    # This captures the full flat-top of a single stripe so we suppress it
    # entirely, not just ±10px.
    flat_thresh = peak1 * 0.80
    l, r = idx1, idx1
    while l > 0 and col[l - 1] >= flat_thresh:
        l -= 1
    while r < len(col) - 1 and col[r + 1] >= flat_thresh:
        r += 1

    c2 = col.copy()
    c2[max(0, l - 4): min(len(c2), r + 5)] = 0   # +buffer to clear edge ramps

    idx2 = int(np.argmax(c2))
    peak2 = c2[idx2]

    if peak2 < 0.28 * peak1:
        return None

    spacing = abs(idx1 - idx2)
    if not (12 <= spacing <= 80):
        return None

    x1_abs = xa + min(idx1, idx2)
    x2_abs = xa + max(idx1, idx2)

    def vcov(xc, band=10):
        xa2 = max(0, xc - band)
        xb2 = min(w, xc + band)
        s = binary[ya:, xa2:xb2]
        return float(np.mean(np.any(s > 0, axis=1))) if s.size else 0.0

    c1v = vcov(x1_abs)
    c2v = vcov(x2_abs)

    if _DEBUG_LANES:
        print(f"    _detect_double_solid: spacing={spacing} peak1={peak1:.1f} "
              f"peak2={peak2:.1f} cov1={c1v:.2f} cov2={c2v:.2f}")

    if c1v >= 0.35 and c2v >= 0.35:
        conf = min(1.0, (c1v + c2v) / 2.0)
        return "double_solid", conf, {"x1": x1_abs, "x2": x2_abs, "spacing": spacing}
    return None


def _detect_double_solid_cc(binary, x_center, ya, search=90):
    """
    CC-based double solid detection: find two tall, narrow vertical blobs
    side by side in the search band around x_center.
    """
    h, w = binary.shape
    xa = max(0, x_center - search)
    xb = min(w, x_center + search)
    roi = binary[ya:, xa:xb]
    roi_h = roi.shape[0]

    if roi_h < 20:
        return None

    roi_u8 = (roi > 0).astype(np.uint8) * 255
    _, _, stats, centroids = cv2.connectedComponentsWithStats(roi_u8, connectivity=8)

    stripes = []
    for i in range(1, len(stats)):
        area = stats[i, cv2.CC_STAT_AREA]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        cx = float(centroids[i][0])
        # Must be tall (≥28% of ROI height), taller than wide, and have enough area
        if bh >= roi_h * 0.28 and area >= 30 and (bw == 0 or bh / bw >= 1.2):
            stripes.append({
                "abs_cx": cx + xa,
                "cov": bh / roi_h,
                "area": area,
            })

    if len(stripes) < 2:
        return None

    stripes.sort(key=lambda s: s["abs_cx"])

    best = None
    for i in range(len(stripes) - 1):
        s1, s2 = stripes[i], stripes[i + 1]
        spacing = abs(s2["abs_cx"] - s1["abs_cx"])
        avg_cov = (s1["cov"] + s2["cov"]) / 2
        if 5 <= spacing <= 80 and avg_cov >= 0.28:
            conf = min(1.0, avg_cov + 0.10)
            if best is None or conf > best[1]:
                best = ("double_solid", conf, {
                    "x1": int(s1["abs_cx"]), "x2": int(s2["abs_cx"]),
                    "spacing": int(spacing), "cov": round(avg_cov, 2),
                })

    if _DEBUG_LANES and best:
        print(f"    _detect_double_solid_cc: {best[2]}")

    return best


def classify_marking(binary, x_center):
    """
    Classify the lane marking at x_center.
    Priority: double_solid (Gaussian peaks + CC consensus) → CC single-line analysis.
    """
    h, w = binary.shape
    ya = int(h * 0.18)

    if _DEBUG_LANES:
        print(f"  classify_marking x={x_center}:")

    # 1. Double solid: either detector is sufficient (OR logic).
    #    If both agree, confidence gets a bonus.
    ds_colsum = _detect_double_solid(binary, x_center, ya)
    ds_cc     = _detect_double_solid_cc(binary, x_center, ya)

    if ds_colsum is not None and ds_cc is not None:
        conf = min(1.0, (ds_colsum[1] + ds_cc[1]) / 2.0 + 0.05)
        if _DEBUG_LANES:
            print(f"    → double_solid (both) conf={conf:.2f}")
        return "double_solid", conf, {**ds_colsum[2], "cc": True}

    if ds_colsum is not None:
        if _DEBUG_LANES:
            print(f"    → double_solid (colsum) conf={ds_colsum[1]:.2f}")
        return ds_colsum  # ("double_solid", conf, meta)

    if ds_cc is not None:
        if _DEBUG_LANES:
            print(f"    → double_solid (cc) conf={ds_cc[1]:.2f}")
        return ds_cc

    # 2. Single-line CC classification on a ±50px strip
    sw = 50
    xa = max(0, x_center - sw)
    xb = min(w, x_center + sw)
    strip = (binary[ya:, xa:xb] > 0).astype(np.uint8) * 255

    t, c, meta = _classify_cc(strip, label=f"[{x_center}]")
    if _DEBUG_LANES:
        print(f"    → {t} conf={c:.2f} meta={meta}")
    return t, c, meta


class LaneLines:
    """Class containing information about detected lane lines."""

    def __init__(self):
        self.left_fit = None
        self.right_fit = None
        self.binary = None
        self.nonzero = None
        self.nonzerox = None
        self.nonzeroy = None
        self.clear_visibility = True
        self.dir = []

        self.left_marking = ("unknown", 0.0)
        self.right_marking = ("unknown", 0.0)

        # History for temporal smoothing
        self.left_hist = []
        self.right_hist = []

        # Short-term memory to hold last confident result
        self.left_last_good = ("unknown", 0.0)
        self.right_last_good = ("unknown", 0.0)
        self.left_last_good_age = 999
        self.right_last_good_age = 999

        # ────────────────────────────────────────────────
        # LEFT double_solid lock — holds double_solid for 4 s at high conf
        self.left_double_lock = None
        self.left_double_lock_until = 0.0
        self.LEFT_LOCK_SECONDS = 4.0
        self.LEFT_LOCK_MIN_CONF = 0.75

        # RIGHT double_solid lock (mirrors left)
        self.right_double_lock = None
        self.right_double_lock_until = 0.0

        # General marking lock — holds ANY type for 4 s when conf ≥ 0.75
        self.left_gen_lock = ("unknown", 0.0)
        self.left_gen_lock_until = 0.0
        self.right_gen_lock = ("unknown", 0.0)
        self.right_gen_lock_until = 0.0
        self.GEN_LOCK_SECONDS = 4.0
        self.GEN_LOCK_MIN_CONF = 0.75

        # ────────────────────────────────────────────────
        # Violation detection state
        self.violation_active = False          # left violation in progress
        self.violation_start_time = 0.0
        self.violation_last_emit_time = 0.0
        self.right_violation_active = False    # right violation in progress
        self.right_violation_start_time = 0.0
        self.right_violation_last_emit_time = 0.0

        self.VIOLATION_CONFIRM_SECONDS = 0.5   # must be crossed for 0.5s
        self.VIOLATION_COOLDOWN_SECONDS = 2.0  # don't report again for 2s
        self.CROSS_MARGIN_PX = 25              # how far past the line counts as crossed
        self.violation_event = None            # latest violation dict for server
        self.violation_display_until = 0.0    # time until warning is shown
        self.VIOLATION_DISPLAY_SECONDS = 6.0  # how long to show warning
        # ────────────────────────────────────────────────
        self.FORCE_TEST_VIOLATION = os.getenv("FORCE_TEST_VIOLATION", "0") == "1"
        self._test_start_time = None
        self._test_triggered = False

        # Load icons
        self.left_curve_img = mpimg.imread('left_turn.png')
        self.right_curve_img = mpimg.imread('right_turn.png')
        self.keep_straight_img = mpimg.imread('straight.png')

        self.left_curve_img = cv2.normalize(
            src=self.left_curve_img, dst=None, alpha=0, beta=255,
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        self.right_curve_img = cv2.normalize(
            src=self.right_curve_img, dst=None, alpha=0, beta=255,
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        self.keep_straight_img = cv2.normalize(
            src=self.keep_straight_img, dst=None, alpha=0, beta=255,
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

        self.nwindows = 9
        self.margin = 120    # wider search window for compressed frames
        self.minpix = 20     # lower minimum to keep tracking thin lane lines

    def reset_stream_state(self):
        """Reset short-lived violation/test state between independent streams."""
        self.violation_active = False
        self.violation_start_time = 0.0
        self.right_violation_active = False
        self.right_violation_start_time = 0.0
        self.violation_event = None
        self.violation_display_until = 0.0
        self._test_start_time = None
        self._test_triggered = False

    def apply_left_double_lock(self):
        """Hold left marking as double_solid for LEFT_LOCK_SECONDS if conf >= LEFT_LOCK_MIN_CONF."""
        t, c = self.left_marking
        now = time.time()

        # If strong detection -> refresh lock window
        if t == "double_solid" and c >= self.LEFT_LOCK_MIN_CONF:
            self.left_double_lock = (t, c)
            self.left_double_lock_until = now + self.LEFT_LOCK_SECONDS
            return

        # If weak now but lock is still active -> keep locked
        if self.left_double_lock is not None and now <= self.left_double_lock_until:
            self.left_marking = self.left_double_lock
        else:
            # lock expired
            self.left_double_lock = None
            self.left_double_lock_until = 0.0

    def apply_right_double_lock(self):
        """Mirror of apply_left_double_lock for the right lane marking."""
        t, c = self.right_marking
        now = time.time()

        if t == "double_solid" and c >= self.LEFT_LOCK_MIN_CONF:
            self.right_double_lock = (t, c)
            self.right_double_lock_until = now + self.LEFT_LOCK_SECONDS
            return

        if self.right_double_lock is not None and now <= self.right_double_lock_until:
            self.right_marking = self.right_double_lock
        else:
            self.right_double_lock = None
            self.right_double_lock_until = 0.0

    def apply_general_lock(self):
        """
        For any lane side: if marking was detected at ≥ GEN_LOCK_MIN_CONF,
        hold that result for GEN_LOCK_SECONDS even when subsequent frames
        return weak or unknown detections.
        """
        now = time.time()
        for marking_attr, lock_attr, until_attr in (
            ("left_marking",  "left_gen_lock",  "left_gen_lock_until"),
            ("right_marking", "right_gen_lock", "right_gen_lock_until"),
        ):
            t, c = getattr(self, marking_attr)
            if c >= self.GEN_LOCK_MIN_CONF and t != "unknown":
                setattr(self, lock_attr, (t, c))
                setattr(self, until_attr, now + self.GEN_LOCK_SECONDS)
            else:
                lock_val = getattr(self, lock_attr)
                if lock_val[0] != "unknown" and now <= getattr(self, until_attr):
                    setattr(self, marking_attr, lock_val)

    def update_violation_left(self, w, left_line_x, y_eval):
        """
        Detect LEFT double_solid crossing violation (time-confirmed + cooldown).
        Rule: if left_marking is double_solid and vehicle center goes left of left boundary.
        """
        now = time.time()
        vehicle_x = w // 2

        lt, lc = self.left_marking

        # vehicle crossed the left boundary by margin
        crossed_left = (vehicle_x < (left_line_x - self.CROSS_MARGIN_PX))

        # enforce only when left boundary is (locked) double solid
        enforce = (lt == "double_solid" and lc >= 0.75)

        if enforce and crossed_left:
            # start/continue timing
            if not self.violation_active:
                self.violation_active = True
                self.violation_start_time = now

            # confirm it stayed crossed for some time
            if (now - self.violation_start_time) >= self.VIOLATION_CONFIRM_SECONDS:
                # cooldown to avoid spamming
                if (now - self.violation_last_emit_time) >= self.VIOLATION_COOLDOWN_SECONDS:
                    self.violation_last_emit_time = now
                    self.violation_event = {
                        "type": "DOUBLE_SOLID_CROSS_LEFT",
                        "time": now,
                        "left_type": lt,
                        "left_conf": float(lc),
                        "vehicle_x": int(vehicle_x),
                        "left_line_x": int(left_line_x),
                        "y_eval": int(y_eval),
                        "margin_px": int(self.CROSS_MARGIN_PX),
                    }
                    self.violation_display_until = now + self.VIOLATION_DISPLAY_SECONDS
        else:
            # reset if not crossed / not enforced
            self.violation_active = False
            self.violation_start_time = 0.0

    def update_violation_right(self, w, right_line_x, y_eval):
        """
        Detect RIGHT double_solid crossing violation (time-confirmed + cooldown).
        Rule: if right_marking is double_solid and vehicle center goes right of right boundary.
        """
        now = time.time()
        vehicle_x = w // 2

        rt, rc = self.right_marking

        crossed_right = (vehicle_x > (right_line_x + self.CROSS_MARGIN_PX))
        enforce = (rt == "double_solid" and rc >= 0.75)

        if enforce and crossed_right:
            if not self.right_violation_active:
                self.right_violation_active = True
                self.right_violation_start_time = now

            if (now - self.right_violation_start_time) >= self.VIOLATION_CONFIRM_SECONDS:
                if (now - self.right_violation_last_emit_time) >= self.VIOLATION_COOLDOWN_SECONDS:
                    self.right_violation_last_emit_time = now
                    self.violation_event = {
                        "type": "DOUBLE_SOLID_CROSS_RIGHT",
                        "time": now,
                        "right_type": rt,
                        "right_conf": float(rc),
                        "vehicle_x": int(vehicle_x),
                        "right_line_x": int(right_line_x),
                        "y_eval": int(y_eval),
                        "margin_px": int(self.CROSS_MARGIN_PX),
                    }
                    self.violation_display_until = now + self.VIOLATION_DISPLAY_SECONDS
        else:
            self.right_violation_active = False
            self.right_violation_start_time = 0.0

    def forward(self, img):
        self.extract_features(img)
        return self.fit_poly(img)

    def pixels_in_window(self, center, margin, height):
        topleft = (center[0] - margin, center[1] - height // 2)
        bottomright = (center[0] + margin, center[1] + height // 2)

        condx = (topleft[0] <= self.nonzerox) & (self.nonzerox <= bottomright[0])
        condy = (topleft[1] <= self.nonzeroy) & (self.nonzeroy <= bottomright[1])
        return self.nonzerox[condx & condy], self.nonzeroy[condx & condy]

    def extract_features(self, img):
        self.img = img
        self.window_height = int(img.shape[0] // self.nwindows)

        self.nonzero = img.nonzero()
        self.nonzerox = np.array(self.nonzero[1])
        self.nonzeroy = np.array(self.nonzero[0])

    def find_lane_pixels(self, img):
        assert len(img.shape) == 2

        out_img = np.dstack((img, img, img))

        histogram = hist(img)
        midpoint = histogram.shape[0] // 2
        leftx_base = int(np.argmax(histogram[:midpoint]))
        rightx_base = int(np.argmax(histogram[midpoint:]) + midpoint)

        leftx_current = leftx_base
        rightx_current = rightx_base
        y_current = img.shape[0] + self.window_height // 2

        leftx, lefty, rightx, righty = [], [], [], []

        for _ in range(self.nwindows):
            y_current -= self.window_height
            center_left = (leftx_current, y_current)
            center_right = (rightx_current, y_current)

            good_left_x, good_left_y = self.pixels_in_window(center_left, self.margin, self.window_height)
            good_right_x, good_right_y = self.pixels_in_window(center_right, self.margin, self.window_height)

            leftx.extend(good_left_x)
            lefty.extend(good_left_y)
            rightx.extend(good_right_x)
            righty.extend(good_right_y)

            if len(good_left_x) > self.minpix:
                leftx_current = int(np.mean(good_left_x))
            if len(good_right_x) > self.minpix:
                rightx_current = int(np.mean(good_right_x))

        return leftx, lefty, rightx, righty, out_img

    def fit_poly(self, img):
        
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(img)

        if len(lefty) > 400:
            self.left_fit = np.polyfit(lefty, leftx, 2)
        if len(righty) > 400:
            self.right_fit = np.polyfit(righty, rightx, 2)

        if self.left_fit is None or self.right_fit is None:
            self.left_marking = ("unknown", 0.0)
            self.right_marking = ("unknown", 0.0)
            return out_img

        if _DEBUG_LANES:
            print(f"[fit_poly] left_x={left_x} right_x={right_x} "
                  f"binary_fill={np.mean(img > 0):.3f}")

        maxy = img.shape[0] - 1
        miny = img.shape[0] // 3

        if len(lefty):
            maxy = max(maxy, int(np.max(lefty)))
            miny = min(miny, int(np.min(lefty)))

        if len(righty):
            maxy = max(maxy, int(np.max(righty)))
            miny = min(miny, int(np.min(righty)))

        ploty = np.linspace(miny, maxy, img.shape[0])

        left_fitx = self.left_fit[0] * ploty**2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty**2 + self.right_fit[1] * ploty + self.right_fit[2]

        for i, y in enumerate(ploty):
            l = int(left_fitx[i])
            r = int(right_fitx[i])
            yy = int(y)
            cv2.line(out_img, (l, yy), (r, yy), (0, 255, 0), 2)

        _ = self.measure_curvature()

        h = img.shape[0]
        y_eval = h - 20

        left_x = int(self.left_fit[0] * y_eval**2 + self.left_fit[1] * y_eval + self.left_fit[2])
        right_x = int(self.right_fit[0] * y_eval**2 + self.right_fit[1] * y_eval + self.right_fit[2])

        clean = clean_binary(img)

        left_type, left_conf, _ = classify_marking(clean, left_x)
        right_type, right_conf, _ = classify_marking(clean, right_x)

        self.left_marking = (left_type, float(left_conf))
        self.right_marking = (right_type, float(right_conf))

        # Store history (majority vote smoothing)
        self.left_hist.append(self.left_marking)
        self.right_hist.append(self.right_marking)

        if len(self.left_hist) > 10:
            self.left_hist.pop(0)
        if len(self.right_hist) > 10:
            self.right_hist.pop(0)

        def stable_label(hist):
            """
            Confidence-weighted majority vote.
            Each frame's vote is weighted by its classification confidence,
            so a single high-confidence reading outweighs several uncertain ones.
            Frames below 0.15 confidence (noise) are excluded entirely.
            """
            scores = {}
            total_weight = 0.0
            for t, c in hist:
                if t == "unknown" or c < 0.15:
                    continue
                scores[t] = scores.get(t, 0.0) + c
                total_weight += c
            if not scores:
                return "unknown", 0.0
            best = max(scores, key=scores.get)
            # Normalised confidence = fraction of total weight this type received
            conf = scores[best] / max(total_weight, 1e-9)
            return best, round(conf, 3)

        self.left_marking = stable_label(self.left_hist)
        self.right_marking = stable_label(self.right_hist)

        # Hold last good result if current is weak / unknown
        def hold_last_good(current, last_good, age, max_age=10):
            t, c = current
            if c >= 0.20 and t != "unknown":
                return (t, c), 0
            if age < max_age and last_good[0] != "unknown":
                return last_good, age + 1
            return current, age + 1

        self.left_marking, self.left_last_good_age = hold_last_good(
            self.left_marking, self.left_last_good, self.left_last_good_age
        )
        self.right_marking, self.right_last_good_age = hold_last_good(
            self.right_marking, self.right_last_good, self.right_last_good_age
        )

        # Update last good trackers
        if self.left_marking[0] != "unknown" and self.left_marking[1] >= 0.20:
            self.left_last_good = self.left_marking
        if self.right_marking[0] != "unknown" and self.right_marking[1] >= 0.20:
            self.right_last_good = self.right_marking

        # ────────────────────────────────────────────────────────────────
        # Apply locks: double_solid-specific first, then general 0.75+ hold
        self.apply_left_double_lock()
        self.apply_right_double_lock()
        self.apply_general_lock()

        # ===== TEMP TEST: FORCE VIOLATION AFTER 5 SECONDS =====
        # Enable with: export FORCE_TEST_VIOLATION=1
        if self.FORCE_TEST_VIOLATION:
            now = time.time()
            if self._test_start_time is None:
                self._test_start_time = now
            # Trigger once after 5 seconds per stream/session.
            if (not self._test_triggered) and (now - self._test_start_time) > 5:
                print("TEST: forcing violation event")
                self.violation_event = {
                    "type": "TEST_FORCED_VIOLATION",
                    "time": now
                }
                self.violation_display_until = now + self.VIOLATION_DISPLAY_SECONDS
                self._test_triggered = True
        # =======================================================

        # Update violation state (LEFT and RIGHT)
        w = img.shape[1]
        self.update_violation_left(w=w, left_line_x=left_x, y_eval=y_eval)
        if self.violation_event is None:
            self.update_violation_right(w=w, right_line_x=right_x, y_eval=y_eval)
        # ────────────────────────────────────────────────────────────────

        return out_img

    def plot(self, out_img):
        np.set_printoptions(precision=6, suppress=True)

        if self.left_fit is None or self.right_fit is None:
            return out_img

        lR, rR, pos = self.measure_curvature()

        value = self.left_fit[0] if abs(self.left_fit[0]) > abs(self.right_fit[0]) else self.right_fit[0]

        if abs(value) <= 0.00015:
            self.dir.append('F')
        elif value < 0:
            self.dir.append('L')
        else:
            self.dir.append('R')

        if len(self.dir) > 10:
            self.dir.pop(0)

        W = 400
        H = 500
        widget = np.copy(out_img[:H, :W])
        widget //= 2
        widget[0, :] = [0, 0, 255]
        widget[-1, :] = [0, 0, 255]
        widget[:, 0] = [0, 0, 255]
        widget[:, -1] = [0, 0, 255]
        out_img[:H, :W] = widget

        direction = max(set(self.dir), key=self.dir.count)
        msg = "Keep Straight Ahead"
        curvature_msg = "Curvature = {:.0f} m".format(min(lR, rR))

        if direction == 'L':
            y, x = self.left_curve_img[:, :, 3].nonzero()
            out_img[y, x - 100 + W // 2] = self.left_curve_img[y, x, :3]
            msg = "Left Curve Ahead"
        if direction == 'R':
            y, x = self.right_curve_img[:, :, 3].nonzero()
            out_img[y, x - 100 + W // 2] = self.right_curve_img[y, x, :3]
            msg = "Right Curve Ahead"
        if direction == 'F':
            y, x = self.keep_straight_img[:, :, 3].nonzero()
            out_img[y, x - 100 + W // 2] = self.keep_straight_img[y, x, :3]

        cv2.putText(out_img, msg, org=(10, 240),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(255, 255, 255), thickness=2)

        if direction in 'LR':
            cv2.putText(out_img, curvature_msg, org=(10, 280),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(255, 255, 255), thickness=2)

        cv2.putText(out_img, "Good Lane Keeping", org=(10, 400),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.2,
                    color=(0, 255, 0), thickness=2)

        cv2.putText(out_img, "Vehicle is {:.2f} m away from center".format(pos),
                    org=(10, 450),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.66,
                    color=(255, 255, 255), thickness=2)

        lt, lc = self.left_marking
        rt, rc = self.right_marking

        cv2.putText(out_img, f"LEFT:  {lt} ({lc:.2f})", (10, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(out_img, f"RIGHT: {rt} ({rc:.2f})", (10, 355),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Show violation warning on overlay (optional)
        if time.time() <= self.violation_display_until:
            cv2.putText(out_img, "VIOLATION: DOUBLE SOLID CROSSING!",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        return out_img

    def measure_curvature(self):
        if self.left_fit is None or self.right_fit is None:
            return 0.0, 0.0, 0.0

        ym = 30 / 720
        xm = 3.7 / 700

        left_fit = self.left_fit.copy()
        right_fit = self.right_fit.copy()
        y_eval = 700 * ym

        left_curveR = ((1 + (2 * left_fit[0] * y_eval + left_fit[1])**2)**1.5) / np.absolute(2 * left_fit[0])
        right_curveR = ((1 + (2 * right_fit[0] * y_eval + right_fit[1])**2)**1.5) / np.absolute(2 * right_fit[0])

        xl = np.dot(self.left_fit, [700**2, 700, 1])
        xr = np.dot(self.right_fit, [700**2, 700, 1])
        pos = (1280 // 2 - (xl + xr) // 2) * xm

        return float(left_curveR), float(right_curveR), float(pos)
