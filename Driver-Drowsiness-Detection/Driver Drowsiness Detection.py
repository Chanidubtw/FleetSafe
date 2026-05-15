#!/usr/bin/env python
import argparse
import platform
import shutil
import subprocess
import threading
import time

import cv2
import dlib
import numpy as np
from imutils import face_utils
from imutils.video import VideoStream

from EAR import eye_aspect_ratio
from HeadPose import getHeadTiltAndCoords
from MAR import mouth_aspect_ratio


class ContinuousBeep:
    def __init__(self, enabled=True, sound_path=None):
        self.enabled = enabled
        self.sound_path = sound_path
        self._running = False
        self._thread = None
        self._proc = None
        self._lock = threading.Lock()
        self._player_mode = self._resolve_player_mode()

    def _resolve_player_mode(self):
        if not self.enabled:
            return "none"

        system = platform.system().lower()
        if system == "darwin" and shutil.which("afplay"):
            if self.sound_path:
                return "afplay"
            default_path = "/System/Library/Sounds/Submarine.aiff"
            self.sound_path = default_path
            return "afplay"
        if system == "windows":
            return "winsound"
        return "terminal_bell"

    def _run(self):
        while True:
            with self._lock:
                if not self._running:
                    break

            if self._player_mode == "afplay":
                with self._lock:
                    self._proc = subprocess.Popen(["afplay", self.sound_path])
                while True:
                    time.sleep(0.03)
                    with self._lock:
                        should_stop = not self._running
                        proc = self._proc
                    if proc.poll() is not None:
                        break
                    if should_stop:
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                        return
            elif self._player_mode == "winsound":
                try:
                    import winsound

                    winsound.Beep(1400, 200)
                except Exception:
                    pass
                time.sleep(0.1)
            elif self._player_mode == "terminal_bell":
                print("\a", end="", flush=True)
                time.sleep(0.4)
            else:
                break

        with self._lock:
            self._proc = None

    def start(self):
        with self._lock:
            if not self.enabled or self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        with self._lock:
            self._running = False
            proc = self._proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass


def parse_args():
    parser = argparse.ArgumentParser(description="Driver drowsiness detection")
    parser.add_argument(
        "--predictor",
        default="./dlib_shape_predictor/shape_predictor_68_face_landmarks.dat",
        help="Path to dlib 68 landmarks predictor file",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=-1,
        help="Camera index. Use -1 to auto-detect.",
    )
    parser.add_argument(
        "--max-camera-index",
        type=int,
        default=5,
        help="Max camera index to probe when auto-detecting.",
    )
    parser.add_argument("--width", type=int, default=1024, help="Frame width")
    parser.add_argument("--height", type=int, default=576, help="Frame height")
    parser.add_argument("--eye-ar-thresh", type=float, default=0.25)
    parser.add_argument("--eye-consec-frames", type=int, default=3)
    parser.add_argument("--eye-drowsy-frames", type=int, default=60)
    parser.add_argument("--mouth-ar-thresh", type=float, default=0.79)
    parser.add_argument(
        "--alarm-sound",
        default="/System/Library/Sounds/Submarine.aiff",
        help="Path to sound file for macOS afplay mode.",
    )
    parser.add_argument(
        "--no-sound",
        action="store_true",
        help="Disable alarm sound completely.",
    )
    parser.add_argument(
        "--show-fps",
        action="store_true",
        help="Overlay estimated FPS.",
    )
    return parser.parse_args()


def find_camera_index(explicit_index, max_index):
    if explicit_index >= 0:
        cap = cv2.VideoCapture(explicit_index)
        ok = cap.isOpened()
        cap.release()
        if not ok:
            raise RuntimeError(
                f"Camera index {explicit_index} is not available. "
                "Check camera permissions and close other camera apps."
            )
        return explicit_index

    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
        cap.release()
    raise RuntimeError(
        "No working camera found. Check camera permissions and close other camera apps."
    )


def main():
    args = parse_args()

    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.predictor)

    print("[INFO] initializing camera...")
    camera_index = find_camera_index(args.camera_index, args.max_camera_index)
    print(f"[INFO] using camera index {camera_index}")

    beeper = ContinuousBeep(enabled=not args.no_sound, sound_path=args.alarm_sound)
    vs = VideoStream(src=camera_index).start()
    time.sleep(1.5)

    image_points = np.array(
        [
            (359, 391),  # Nose tip 34
            (399, 561),  # Chin 9
            (337, 297),  # Left eye left corner 37
            (513, 301),  # Right eye right corner 46
            (345, 465),  # Left Mouth corner 49
            (453, 469),  # Right mouth corner 55
        ],
        dtype="double",
    )

    (l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (m_start, m_end) = (49, 68)

    counter = 0
    frame_counter = 0
    fps_started_at = time.time()
    fps_value = 0.0

    try:
        while True:
            frame = vs.read()
            if frame is None:
                time.sleep(0.02)
                continue

            frame = cv2.resize(frame, (args.width, args.height))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            size = gray.shape
            rects = detector(gray, 0)

            if rects:
                cv2.putText(
                    frame,
                    f"{len(rects)} face(s) found",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

            for rect in rects:
                (b_x, b_y, b_w, b_h) = face_utils.rect_to_bb(rect)
                cv2.rectangle(frame, (b_x, b_y), (b_x + b_w, b_y + b_h), (0, 255, 0), 1)

                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                left_eye = shape[l_start:l_end]
                right_eye = shape[r_start:r_end]
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0

                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)
                cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

                cv2.putText(
                    frame,
                    f"EAR: {ear:.2f}",
                    (args.width - 370, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

                if ear < args.eye_ar_thresh:
                    counter += 1
                    if counter >= args.eye_consec_frames:
                        cv2.putText(
                            frame,
                            "Eyes Closed!",
                            (args.width - 220, 55),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )
                    if counter >= args.eye_drowsy_frames:
                        cv2.putText(
                            frame,
                            "DROWSINESS ALERT!",
                            (args.width // 2 - 170, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 0, 255),
                            3,
                        )
                        beeper.start()
                else:
                    counter = 0
                    beeper.stop()

                mouth = shape[m_start:m_end]
                mar = mouth_aspect_ratio(mouth)
                mouth_hull = cv2.convexHull(mouth)
                cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 0), 1)
                cv2.putText(
                    frame,
                    f"MAR: {mar:.2f}",
                    (args.width - 180, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

                if mar > args.mouth_ar_thresh:
                    cv2.putText(
                        frame,
                        "Yawning!",
                        (args.width - 170, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                for (i, (x, y)) in enumerate(shape):
                    if i == 33:
                        image_points[0] = np.array([x, y], dtype="double")
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    elif i == 8:
                        image_points[1] = np.array([x, y], dtype="double")
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    elif i == 36:
                        image_points[2] = np.array([x, y], dtype="double")
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    elif i == 45:
                        image_points[3] = np.array([x, y], dtype="double")
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    elif i == 48:
                        image_points[4] = np.array([x, y], dtype="double")
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    elif i == 54:
                        image_points[5] = np.array([x, y], dtype="double")
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    else:
                        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

                for p in image_points:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

                try:
                    (head_tilt_degree, start_point, end_point, end_point_alt) = (
                        getHeadTiltAndCoords(size, image_points, args.height)
                    )
                    cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
                    cv2.line(frame, start_point, end_point_alt, (0, 0, 255), 2)
                    if head_tilt_degree is not None:
                        cv2.putText(
                            frame,
                            f"Head Tilt Degree: {head_tilt_degree[0]}",
                            (170, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            2,
                        )
                except cv2.error:
                    pass

            if args.show_fps:
                frame_counter += 1
                elapsed = time.time() - fps_started_at
                if elapsed >= 1.0:
                    fps_value = frame_counter / elapsed
                    frame_counter = 0
                    fps_started_at = time.time()
                cv2.putText(
                    frame,
                    f"FPS: {fps_value:.1f}",
                    (10, args.height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

            cv2.imshow("Frame", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        vs.stop()
        beeper.stop()


if __name__ == "__main__":
    main()
