"""Reliable OpenCV webcam capture."""
from __future__ import annotations

import cv2
import numpy as np


class Camera:
    def __init__(self, index: int, width: int, height: int) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.capture: cv2.VideoCapture | None = None
        self.backend_name = ""
        self.failed_reads = 0

    def open(self) -> bool:
        backends = [("ANY", cv2.CAP_ANY)]
        if hasattr(cv2, "CAP_DSHOW"):
            backends.append(("DSHOW", cv2.CAP_DSHOW))
        for name, backend in backends:
            capture = cv2.VideoCapture(self.index, backend)
            if not capture.isOpened():
                capture.release()
                continue
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            for _ in range(8):
                ok, frame = capture.read()
                if ok and frame is not None:
                    self.capture = capture
                    self.backend_name = name
                    self.failed_reads = 0
                    return True
            capture.release()
        self.capture = None
        return False

    def read(self) -> np.ndarray | None:
        if self.capture is None:
            return None
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.failed_reads += 1
            return None
        self.failed_reads = 0
        return cv2.flip(frame, 1)

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
