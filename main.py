"""
Lane Lines Detection pipeline

Usage:
    main.py [--video] INPUT_PATH OUTPUT_PATH

Options:
    -h --help   show this screen
    --video     process video file instead of image
"""

import numpy as np
import matplotlib.image as mpimg
import cv2
try:
    from docopt import docopt
    from moviepy.editor import VideoFileClip
except ImportError:
    # These are only needed for CLI video processing
    pass


from CameraCalibration import CameraCalibration
from Thresholding import Thresholding
from PerspectiveTransformation import PerspectiveTransformation
from LaneLines import LaneLines


def ensure_bgr_and_size(overlay_img, base_img):
    """
    Make overlay_img compatible with base_img for cv2.addWeighted:
    - same width/height
    - 3 channels (BGR/RGB compatible for addWeighted)
    """
    # Ensure overlay is numpy array
    overlay = np.array(overlay_img)

    # Convert grayscale -> BGR
    if len(overlay.shape) == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    # Some pipelines return RGBA; drop alpha if present
    if len(overlay.shape) == 3 and overlay.shape[2] == 4:
        overlay = overlay[:, :, :3]

    # Resize to match base
    if overlay.shape[:2] != base_img.shape[:2]:
        overlay = cv2.resize(overlay, (base_img.shape[1], base_img.shape[0]))

    return overlay


class FindLaneLines:
    def __init__(self):
        self.calibration = CameraCalibration('camera_cal', 9, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()

    def forward(self, frame):
        # Keep original frame for overlay
        out_img = np.copy(frame)

        # Skip camera undistortion — calibration matrices are camera-specific
        # and will distort frames from a different camera/video source.
        img = frame
        img = self.transform.forward(img)
        img = self.thresholding.forward(img)
        img = self.lanelines.forward(img)
        img = self.transform.backward(img)

        # FIX: make sure overlay "img" matches out_img shape/channels
        img = ensure_bgr_and_size(img, out_img)

        # Blend overlay with original
        out_img = cv2.addWeighted(out_img, 1.0, img, 0.6, 0)

        # Draw widget + text (lane marking labels etc.)
        out_img = self.lanelines.plot(out_img)
        return out_img

    def process_image(self, input_path, output_path):
        img = mpimg.imread(input_path)

        # mpimg can return float 0..1; convert to uint8 for cv2 operations
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        out_img = self.forward(img)

        # Save using mpimg (expects RGB). out_img is OK as numpy array.
        mpimg.imsave(output_path, out_img)

    def process_video(self, input_path, output_path):
        clip = VideoFileClip(input_path)
        out_clip = clip.fl_image(self.forward)
        out_clip.write_videofile(output_path, audio=False)
        clip.close()
        out_clip.close()


def main():
    args = docopt(__doc__)
    input_path = args['INPUT_PATH']
    output_path = args['OUTPUT_PATH']

    findLaneLines = FindLaneLines()

    if args['--video']:
        findLaneLines.process_video(input_path, output_path)
    else:
        findLaneLines.process_image(input_path, output_path)


if __name__ == "__main__":
    main()
