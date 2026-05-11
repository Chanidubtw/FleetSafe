import cv2
from collections import deque
import threading


class _Recording:
    def __init__(self, frames, target_frames):
        self.frames = frames
        self.target_frames = target_frames
        self.done = threading.Event()


class FrameBuffer:
    def __init__(self, fps=20, seconds=8, pre_seconds=2):
        self.max_frames = max(1, int(fps * seconds))
        self.buffer = deque(maxlen=self.max_frames)
        self.fps = fps
        self.seconds = seconds
        self.pre_seconds = max(0, pre_seconds)
        self._lock = threading.Lock()
        self._recordings = []

    def add_frame(self, frame):
        """
        Add a frame to rolling buffer.
        """
        frame_copy = frame.copy()
        with self._lock:
            self.buffer.append(frame_copy)
            if not self._recordings:
                return
            active = []
            for rec in self._recordings:
                rec.frames.append(frame_copy)
                if len(rec.frames) >= rec.target_frames:
                    rec.done.set()
                else:
                    active.append(rec)
            self._recordings = active

    def start_recording(self):
        """
        Start a new recording. It will include pre-roll frames (if available)
        and then keep collecting frames until `seconds` is reached.
        """
        target_frames = max(1, int(self.fps * self.seconds))
        pre_frames_count = min(int(self.fps * self.pre_seconds), target_frames)

        with self._lock:
            pre_frames = list(self.buffer)[-pre_frames_count:] if pre_frames_count > 0 else []
            rec = _Recording(frames=pre_frames, target_frames=target_frames)
            if len(rec.frames) >= rec.target_frames:
                rec.done.set()
            else:
                self._recordings.append(rec)
            return rec

    def save_recording(self, recording, output_path):
        """
        Save buffered frames as video clip.
        """
        recording.done.wait(timeout=self.seconds + 2)
        frames = recording.frames
        if not frames:
            return None

        h, w = frames[0].shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))

        for f in frames[: recording.target_frames]:
            out.write(f)

        out.release()
        return output_path
