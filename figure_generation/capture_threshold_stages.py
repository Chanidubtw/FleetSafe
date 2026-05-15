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

# ── 1. Load one of your actual test images ──────────────────────────────────
img_path = 'test_images/frame_100.jpg'   # change to any you like
img_rgb  = mpimg.imread(img_path)                         # reads as RGB

# ── 2. Undistort ─────────────────────────────────────────────────────────────
cal  = CameraCalibration('camera_cal', 9, 6)
undist = cal.undistort(img_rgb)

# ── 3. Warp to bird's-eye ────────────────────────────────────────────────────
pt   = PerspectiveTransformation()
warped = pt.forward(undist)                               # still colour RGB

# ── 4. Full binary output (your actual pipeline result) ──────────────────────
thresh = Thresholding()
binary = thresh.forward(warped)                           # 0 / 255 binary

# ── 5. Draw the trapezoid ROI on the undistorted frame ───────────────────────
h, w = undist.shape[:2]
roi_pts = np.array([[
    (int(w*0.44), int(h*0.63)),
    (int(w*0.10), h),
    (int(w*0.92), h),
    (int(w*0.58), int(h*0.63)),
]], dtype=np.int32)
undist_roi = undist.copy()
cv2.polylines(undist_roi, roi_pts, isClosed=True, color=(255, 80, 80), thickness=3)

# ── 6. Plot 3-panel figure ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.patch.set_facecolor('#1a1a2e')
titles = [
    'Stage 1: Undistorted Frame\n(with ROI trapezoid)',
    'Stage 2: Bird\'s-Eye Warp\n(perspective transform)',
    'Stage 3: Combined Binary\n(morphologically cleaned)',
]
images = [undist_roi, warped, binary]
cmaps  = [None, None, 'gray']

for ax, title, im, cm in zip(axes, titles, images, cmaps):
    ax.set_facecolor('#1a1a2e')
    if cm:
        ax.imshow(im, cmap=cm, vmin=0, vmax=255)
    else:
        ax.imshow(im)
    ax.set_title(title, color='white', fontsize=9, pad=6, fontweight='bold')
    ax.axis('off')

plt.suptitle(
    'FleetSafe — Thresholding Pipeline: Stages 1–3 on Real Road Frame',
    color='white', fontsize=11, fontweight='bold', y=1.01
)
plt.tight_layout(pad=0.4)

# ── 7. Save as high-res PNG ──────────────────────────────────────────────────
save_path = 'threshold_pipeline_figure.png'
plt.savefig(save_path, dpi=200, bbox_inches='tight',
            facecolor='#1a1a2e', edgecolor='none')
print(f'Saved: {save_path}')

if _HAS_GUI:
    plt.show()
else:
    print('GUI backend unavailable; figure was saved without opening a window.')
