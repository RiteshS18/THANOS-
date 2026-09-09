"""MediaPipe hand landmark tracking."""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np


class HandTracker:
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

    def __init__(self, model_path: str, max_num_hands: int = 2, detection_confidence: float = 0.55, tracking_confidence: float = 0.55) -> None:
        self.landmark_model = None
        self.results = None
        self.error: str | None = None
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            self.mp = mp
            model_file = Path(model_path)
            if not model_file.exists():
                model_file.parent.mkdir(parents=True, exist_ok=True)
                urlretrieve(self.MODEL_URL, model_file)
            options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(model_file)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=max_num_hands,
                min_hand_detection_confidence=detection_confidence,
                min_hand_presence_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
            )
            self.landmark_model = vision.HandLandmarker.create_from_options(options)
            self.timestamp_ms = 0
        except Exception as exc:
            self.error = f"MediaPipe unavailable: {exc}"

    @property
    def ready(self) -> bool:
        return self.landmark_model is not None

    def process(self, frame: np.ndarray) -> list[tuple[float, float, float]] | None:
        if self.landmark_model is None:
            return None
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            self.timestamp_ms += 33
            self.results = self.landmark_model.detect_for_video(image, self.timestamp_ms)
            if not self.results.hand_landmarks:
                return None
            return [(point.x, point.y, point.z) for point in self.results.hand_landmarks[0]]
        except Exception as exc:
            self.error = f"Hand tracking failed: {exc}"
            return None

    def get_landmarks(self) -> list[tuple[float, float, float]] | None:
        if not self.results or not self.results.hand_landmarks:
            return None
        return [(point.x, point.y, point.z) for point in self.results.hand_landmarks[0]]

    def draw_landmarks(self, frame: np.ndarray) -> np.ndarray:
        if not self.results or not self.results.hand_landmarks:
            return frame
        height, width = frame.shape[:2]
        connections = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                       (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14),
                       (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20))
        for hand in self.results.hand_landmarks:
            points = [(int(point.x * width), int(point.y * height)) for point in hand]
            for start, end in connections:
                cv2.line(frame, points[start], points[end], (80, 220, 120), 2, cv2.LINE_AA)
            for point in points:
                cv2.circle(frame, point, 3, (40, 220, 255), -1, cv2.LINE_AA)
        return frame

    def close(self) -> None:
        if self.landmark_model is not None:
            self.landmark_model.close()
