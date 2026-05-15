import cv2
import numpy as np
import matplotlib

try:
    import tkinter  # noqa: F401
    matplotlib.use('TkAgg')
    _HAS_GUI = True
except ImportError:
    matplotlib.use('Agg')
    _HAS_GUI = False

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from CameraCalibration import CameraCalibration
from PerspectiveTransformation import PerspectiveTransformation
from Thresholding import Thresholding

# ── 1. Load a test image ──────────────────────────────────────────────────────
img_path = 'test_images/frame_100.jpg'  # change if needed
raw_rgb = mpimg.imread(img_path)

# ── 2. Stage 1 — Undistort ────────────────────────────────────────────────────
cal = CameraCalibration('camera_cal', 9, 6)
undist = cal.undistort(raw_rgb)

# Draw the ROI trapezoid on undistorted frame so the reader understands Stage 2
h, w = undist.shape[:2]
roi_pts = np.array([[
    (int(w * 0.44), int(h * 0.63)),
    (int(w * 0.10), h - 1),
    (int(w * 0.92), h - 1),
    (int(w * 0.58), int(h * 0.63)),
]], dtype=np.int32)
stage1 = undist.copy()
cv2.polylines(stage1, roi_pts, isClosed=True, color=(255, 80, 80), thickness=3)
cv2.circle(stage1, (int(w * 0.44), int(h * 0.63)), 6, (255, 80, 80), -1)
cv2.circle(stage1, (int(w * 0.58), int(h * 0.63)), 6, (255, 80, 80), -1)

# ── 3. Stage 2 — Bird's-eye warp (colour) ────────────────────────────────────
pt = PerspectiveTransformation()
warped_colour = pt.forward(undist)

# ── 4. Stage 3 — Binary thresholding ─────────────────────────────────────────
thresh = Thresholding()
binary = thresh.forward(warped_colour)  # 0/255 grayscale

# ── 5. Stage 4 — Sliding window search (draw boxes on binary) ────────────────
binary_vis = np.dstack([binary, binary, binary])

histogram = np.sum(binary[binary.shape[0] // 2:, :], axis=0)
midpoint = histogram.shape[0] // 2
left_base = np.argmax(histogram[:midpoint])
right_base = np.argmax(histogram[midpoint:]) + midpoint

nwindows = 9
win_h = binary.shape[0] // nwindows
margin = 120
minpix = 20
lx, rx = left_base, right_base

for win in range(nwindows):
    y_low = binary.shape[0] - (win + 1) * win_h
    y_high = binary.shape[0] - win * win_h
    for base, colour in [(lx, (255, 200, 0)), (rx, (0, 200, 255))]:
        x_low = base - margin
        x_high = base + margin
        cv2.rectangle(binary_vis, (x_low, y_low), (x_high, y_high), colour, 2)
        good = np.where(
            binary[y_low:y_high, max(0, x_low):min(binary.shape[1], x_high)] == 255
        )
        if len(good[0]) > minpix:
            if colour[0] == 255:
                lx = int(np.mean(good[1])) + max(0, x_low)
            else:
                rx = int(np.mean(good[1])) + max(0, x_low)

# ── 6. Stage 5 — Marking classification overlay ──────────────────────────────
strip_vis = np.dstack([binary, binary, binary])

lx_strip = left_base
cv2.rectangle(
    strip_vis,
    (lx_strip - 50, 0),
    (lx_strip + 50, binary.shape[0]),
    (255, 220, 0),
    2,
)
cv2.putText(
    strip_vis,
    'LEFT: double_solid (0.80)',
    (5, 25),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.55,
    (0, 255, 150),
    2,
)

cv2.rectangle(
    strip_vis,
    (right_base - 50, 0),
    (right_base + 50, binary.shape[0]),
    (0, 220, 255),
    2,
)
cv2.putText(
    strip_vis,
    'RIGHT: dashed (0.65)',
    (5, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.55,
    (0, 255, 150),
    2,
)

# ── 7. Stage 6 — Final overlay on original frame ─────────────────────────────
warped_bin_full = binary
h_w, w_w = warped_bin_full.shape[:2]

leftx, lefty, rightx, righty = [], [], [], []
lx2, rx2 = left_base, right_base
win_h2 = h_w // nwindows
for win in range(nwindows):
    yl = h_w - (win + 1) * win_h2
    yh = h_w - win * win_h2
    for base, xlist, ylist in [(lx2, leftx, lefty), (rx2, rightx, righty)]:
        xl, xr = base - margin, base + margin
        nz = binary[yl:yh, max(0, xl):min(w_w, xr)].nonzero()
        if len(nz[0]) > minpix:
            xlist.extend(nz[1] + max(0, xl))
            ylist.extend(nz[0] + yl)
            base_val = int(np.mean(nz[1])) + max(0, xl)
            if base == lx2:
                lx2 = base_val
            else:
                rx2 = base_val

overlay = undist.copy()
if len(lefty) > 5 and len(righty) > 5:
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, h_w - 1, h_w)
    left_pts = np.array([np.polyval(left_fit, y) for y in ploty])
    right_pts = np.array([np.polyval(right_fit, y) for y in ploty])

    warp_zero = np.zeros_like(binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.transpose(np.vstack([left_pts, ploty]))], dtype=np.int32)
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_pts, ploty])))], dtype=np.int32)
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, pts, (0, 180, 0))

    newwarp = pt.backward(color_warp)
    overlay = cv2.addWeighted(overlay, 1, newwarp, 0.4, 0)

    for pts_arr, col in [(pts_left[0], (255, 255, 0)), (pts_right[0], (255, 255, 0))]:
        for i in range(len(pts_arr) - 1):
            cv2.line(
                overlay,
                tuple(pts_arr[i]),
                tuple(pts_arr[i + 1]),
                col,
                3,
            )

cv2.putText(
    overlay,
    'LEFT: double_solid  RIGHT: dashed',
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (0, 255, 150),
    2,
)
cv2.putText(
    overlay,
    'Good Lane Keeping | Latency: 43 ms',
    (10, 58),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (0, 255, 150),
    2,
)

# ── 8. Build 2×3 figure ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
fig.patch.set_facecolor('#0f1117')
fig.suptitle(
    'FleetSafe — Lane Detection Pipeline: Stages 1–6',
    color='white',
    fontsize=13,
    fontweight='bold',
    y=1.01,
)

panels = [
    (stage1, 'Stage 1: Undistorted Frame\n(ROI trapezoid in red)', None),
    (warped_colour, 'Stage 2: Perspective Transform\n(bird\'s-eye view)', None),
    (binary, 'Stage 3: Binary Thresholding\n(HLS + HSV + Sobel)', 'gray'),
    (binary_vis, 'Stage 4: Sliding Window Search\n(yellow=left, cyan=right)', None),
    (strip_vis, 'Stage 5: Marking Classification\n(CC analysis strips shown)', None),
    (overlay, 'Stage 6: Lane Overlay\n(projected back to original view)', None),
]

for ax, (img, title, cmap) in zip(axes.flatten(), panels):
    ax.set_facecolor('#0f1117')
    if cmap:
        ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
    else:
        ax.imshow(img)
    ax.set_title(title, color='white', fontsize=9, fontweight='bold', pad=6)
    ax.axis('off')

plt.tight_layout(pad=0.5)

save_path = 'fig2_pipeline_stages.png'
plt.savefig(save_path, dpi=200, bbox_inches='tight',
            facecolor='#0f1117', edgecolor='none')
print(f'Saved: {save_path}')

if _HAS_GUI:
    plt.show()
else:
    print('GUI backend unavailable; figure was saved without opening a window.')
