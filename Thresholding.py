import cv2
import numpy as np


class Thresholding:
    """
    Robust lane thresholding pipeline:
      1. CLAHE contrast enhancement (handles shadowed / low-contrast roads)
      2. White lane detection (HLS)
      3. Yellow lane detection (HSV)
      4. Edge detection (Sobel X)
      5. Morphological cleanup
    """

    def __init__(self):
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _enhance(self, img):
        """Apply CLAHE on the L channel to boost contrast before thresholding."""
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def forward(self, img):
        img = self._enhance(img)

        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        L = hls[:, :, 1]
        S = hls[:, :, 2]
        H = hls[:, :, 0]
        V = hsv[:, :, 2]

        # ── White lanes ────────────────────────────────────────────────
        # Relaxed: L > 180 (was 200) and S < 150 (was 120)
        white = np.zeros_like(L)
        white[(L > 180) & (S < 150)] = 255

        # ── Yellow lanes ───────────────────────────────────────────────
        # Wider hue range (10–45), lower saturation floor (70)
        yellow = np.zeros_like(L)
        yellow[
            (H > 10) & (H < 45) &
            (S > 70) &
            (V > 100)
        ] = 255

        # ── Edges (Sobel X) ────────────────────────────────────────────
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.absolute(sobelx)
        max_val = np.max(sobelx)
        if max_val > 0:
            sobelx_u8 = np.uint8(255 * sobelx / max_val)
        else:
            sobelx_u8 = np.zeros_like(gray, dtype=np.uint8)
        edges = np.zeros_like(sobelx_u8, dtype=np.uint8)
        # Wider range (20–220) catches both light and strong edges
        edges[(sobelx_u8 > 20) & (sobelx_u8 < 220)] = 255

        # ── L-channel bright-area boost ────────────────────────────────
        # Catches bright lane markings that might be missed by color alone
        bright = np.zeros(L.shape, dtype=np.uint8)
        bright[L > 210] = 255

        # ── Combine ────────────────────────────────────────────────────
        binary = cv2.bitwise_or(white, yellow)
        binary = cv2.bitwise_or(binary, edges)
        binary = cv2.bitwise_or(binary, bright)

        # ── Morphological cleanup ──────────────────────────────────────
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)

        return binary
