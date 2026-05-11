import cv2
import numpy as np


class PerspectiveTransformation:
    """Transforms between front-view and bird's-eye view.

    ROI is computed as percentages of the image dimensions so it adapts
    to any camera resolution without manual recalibration.
    """

    # Trapezoid source points as fractions of (width, height)
    # Tune these if the video has an unusual camera angle:
    #   TOP_Y      — horizon line (lower fraction = more sky included)
    #   TOP_LEFT_X / TOP_RIGHT_X — width of trapezoid top
    #   BOT_LEFT_X / BOT_RIGHT_X — width of trapezoid base
    TOP_Y       = 0.63
    TOP_LEFT_X  = 0.44
    TOP_RIGHT_X = 0.58
    BOT_LEFT_X  = 0.10
    BOT_RIGHT_X = 0.92

    def __init__(self):
        self._cached_size = None
        self.M = None
        self.M_inv = None

    def _build(self, w, h):
        src = np.float32([
            (w * self.TOP_LEFT_X,  h * self.TOP_Y),   # top-left
            (w * self.BOT_LEFT_X,  h),                 # bottom-left
            (w * self.BOT_RIGHT_X, h),                 # bottom-right
            (w * self.TOP_RIGHT_X, h * self.TOP_Y),   # top-right
        ])
        dst = np.float32([
            (w * 0.20, 0),
            (w * 0.20, h),
            (w * 0.80, h),
            (w * 0.80, 0),
        ])
        self.M     = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)
        self._cached_size = (w, h)

    def _ensure(self, img):
        h, w = img.shape[:2]
        if self._cached_size != (w, h):
            self._build(w, h)

    def forward(self, img, flags=cv2.INTER_LINEAR):
        self._ensure(img)
        h, w = img.shape[:2]
        return cv2.warpPerspective(img, self.M, (w, h), flags=flags)

    def backward(self, img, flags=cv2.INTER_LINEAR):
        self._ensure(img)
        h, w = img.shape[:2]
        return cv2.warpPerspective(img, self.M_inv, (w, h), flags=flags)
