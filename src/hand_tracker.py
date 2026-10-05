"""
hand_tracker.py — MediaPipe Hands wrapper with landmark utilities.

Uses the modern MediaPipe Tasks API (HandLandmarker) for real-time hand
detection with 21 landmark points. Provides palm center computation,
fingertip extraction, hand bounding box, and orientation angle.
Includes exponential moving average smoothing to reduce landmark jitter.
"""

import math
import os
from collections import namedtuple
from typing import List, Optional, Dict, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np

# ─────────────────────────── Configurable Constants ───────────────────────────
MAX_HANDS = 1                    # Number of hands to detect (1 = single gauntlet)
MIN_DETECTION_CONFIDENCE = 0.6   # Palm detection confidence threshold
MIN_TRACKING_CONFIDENCE = 0.5    # Landmark tracking confidence threshold
SMOOTHING_FACTOR = 0.35          # EMA smoothing (0 = no smoothing, 1 = full lag)

# Path to the hand landmarker model file
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
MODEL_PATH = os.path.join(_PROJECT_DIR, "assets", "hand_landmarker.task")
# ──────────────────────────────────────────────────────────────────────────────

# Named tuple returned for each detected hand
HandData = namedtuple("HandData", [
    "landmarks",          # List of 21 (x, y) pixel-coordinate tuples
    "palm_center",        # (x, y) center of palm
    "hand_bbox",          # (x_min, y_min, x_max, y_max)
    "finger_tips",        # dict: {finger_name: (x, y)}
    "hand_angle",         # Rotation angle in degrees (wrist → middle MCP)
    "handedness",         # "Left" or "Right"
])

# MediaPipe landmark indices
_WRIST = 0
_THUMB_CMC = 1
_THUMB_MCP = 2
_THUMB_IP = 3
_THUMB_TIP = 4
_INDEX_MCP = 5
_INDEX_PIP = 6
_INDEX_DIP = 7
_INDEX_TIP = 8
_MIDDLE_MCP = 9
_MIDDLE_PIP = 10
_MIDDLE_DIP = 11
_MIDDLE_TIP = 12
_RING_MCP = 13
_RING_PIP = 14
_RING_DIP = 15
_RING_TIP = 16
_PINKY_MCP = 17
_PINKY_PIP = 18
_PINKY_DIP = 19
_PINKY_TIP = 20

# Palm landmarks used for center computation
_PALM_LANDMARKS = [_WRIST, _INDEX_MCP, _MIDDLE_MCP, _RING_MCP, _PINKY_MCP]

# Fingertip landmark indices mapped to names
_FINGERTIP_MAP = {
    "thumb":  _THUMB_TIP,
    "index":  _INDEX_TIP,
    "middle": _MIDDLE_TIP,
    "ring":   _RING_TIP,
    "pinky":  _PINKY_TIP,
}


class HandTracker:
    """
    Real-time hand tracker using MediaPipe Tasks API (HandLandmarker).

    Usage:
        tracker = HandTracker()
        while True:
            frame = capture()
            hands = tracker.detect(frame)
            for hand in hands:
                print(hand.palm_center, hand.finger_tips)
        tracker.release()
    """

    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Hand landmarker model not found at: {MODEL_PATH}\n"
                "Download it from: https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task\n"
                "and place it in the assets/ directory."
            )

        # Create HandLandmarker using the Tasks API (VIDEO mode for frame-by-frame)
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MIN_TRACKING_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

        # Frame timestamp counter (required for VIDEO mode)
        self._timestamp_ms = 0

        # Smoothing state: stores previous landmarks per hand index
        self._prev_landmarks: Dict[int, List[Tuple[float, float]]] = {}

    def detect(self, frame: np.ndarray) -> List[HandData]:
        """
        Detect hands in the given BGR frame.

        Args:
            frame: BGR image from cv2.VideoCapture.

        Returns:
            List of HandData namedtuples (one per detected hand).
        """
        h, w, _ = frame.shape

        # Convert BGR → RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image and detect
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp_ms += 33  # ~30fps increment
        results = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        hands = []
        if results.hand_landmarks:
            for idx, hand_lms in enumerate(results.hand_landmarks):
                # Convert normalized landmarks to pixel coordinates
                raw_landmarks = [
                    (int(lm.x * w), int(lm.y * h))
                    for lm in hand_lms
                ]

                # Apply EMA smoothing
                landmarks = self._smooth_landmarks(idx, raw_landmarks)

                # Compute derived data
                palm_center = self._get_palm_center(landmarks)
                hand_bbox = self._get_hand_bbox(landmarks)
                finger_tips = self._get_finger_tips(landmarks)
                hand_angle = self._get_hand_angle(landmarks)

                # Get handedness
                handedness = "Right"
                if results.handedness and idx < len(results.handedness):
                    handedness = results.handedness[idx][0].category_name

                hands.append(HandData(
                    landmarks=landmarks,
                    palm_center=palm_center,
                    hand_bbox=hand_bbox,
                    finger_tips=finger_tips,
                    hand_angle=hand_angle,
                    handedness=handedness,
                ))

        # Clear smoothing state for hands that disappeared
        active_indices = set(range(len(hands)))
        stale = [k for k in self._prev_landmarks if k not in active_indices]
        for k in stale:
            del self._prev_landmarks[k]

        return hands

    def draw_debug(self, frame: np.ndarray, hands: List[HandData]) -> np.ndarray:
        """
        Draw landmark points and connections on the frame for debugging.
        """
        overlay = frame.copy()
        for hand in hands:
            # Draw landmark dots
            for i, (x, y) in enumerate(hand.landmarks):
                color = (0, 255, 0) if i in _PALM_LANDMARKS else (255, 255, 255)
                cv2.circle(overlay, (x, y), 3, color, -1)

            # Draw connections (simple lines between key landmarks)
            connections = [
                (_WRIST, _THUMB_CMC), (_THUMB_CMC, _THUMB_MCP),
                (_THUMB_MCP, _THUMB_IP), (_THUMB_IP, _THUMB_TIP),
                (_WRIST, _INDEX_MCP), (_INDEX_MCP, _INDEX_PIP),
                (_INDEX_PIP, _INDEX_DIP), (_INDEX_DIP, _INDEX_TIP),
                (_WRIST, _MIDDLE_MCP), (_MIDDLE_MCP, _MIDDLE_PIP),
                (_MIDDLE_PIP, _MIDDLE_DIP), (_MIDDLE_DIP, _MIDDLE_TIP),
                (_WRIST, _RING_MCP), (_RING_MCP, _RING_PIP),
                (_RING_PIP, _RING_DIP), (_RING_DIP, _RING_TIP),
                (_WRIST, _PINKY_MCP), (_PINKY_MCP, _PINKY_PIP),
                (_PINKY_PIP, _PINKY_DIP), (_PINKY_DIP, _PINKY_TIP),
                (_INDEX_MCP, _MIDDLE_MCP), (_MIDDLE_MCP, _RING_MCP),
                (_RING_MCP, _PINKY_MCP),
            ]
            for start, end in connections:
                pt1 = hand.landmarks[start]
                pt2 = hand.landmarks[end]
                cv2.line(overlay, pt1, pt2, (0, 200, 200), 1, cv2.LINE_AA)

            # Draw palm center
            cx, cy = hand.palm_center
            cv2.circle(overlay, (cx, cy), 6, (0, 0, 255), -1)
            cv2.circle(overlay, (cx, cy), 8, (255, 255, 255), 1)

            # Draw bounding box
            x1, y1, x2, y2 = hand.hand_bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 200, 0), 1)

            # Draw fingertip labels
            for name, pos in hand.finger_tips.items():
                cv2.putText(overlay, name[0].upper(), (pos[0] + 5, pos[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

        return overlay

    def release(self):
        """Release MediaPipe resources."""
        self._landmarker.close()

    # ──────────────────────── Private Helpers ────────────────────────

    def _smooth_landmarks(
        self, hand_idx: int, raw: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Apply exponential moving average smoothing to landmarks."""
        if hand_idx not in self._prev_landmarks:
            self._prev_landmarks[hand_idx] = list(raw)
            return list(raw)

        prev = self._prev_landmarks[hand_idx]
        smoothed = []
        for (rx, ry), (px, py) in zip(raw, prev):
            sx = int(SMOOTHING_FACTOR * px + (1 - SMOOTHING_FACTOR) * rx)
            sy = int(SMOOTHING_FACTOR * py + (1 - SMOOTHING_FACTOR) * ry)
            smoothed.append((sx, sy))

        self._prev_landmarks[hand_idx] = smoothed
        return smoothed

    @staticmethod
    def _get_palm_center(landmarks: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Compute palm center as average of wrist + 4 MCP joints."""
        xs = [landmarks[i][0] for i in _PALM_LANDMARKS]
        ys = [landmarks[i][1] for i in _PALM_LANDMARKS]
        return (int(np.mean(xs)), int(np.mean(ys)))

    @staticmethod
    def _get_hand_bbox(landmarks: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        """Compute axis-aligned bounding box around all landmarks."""
        xs = [lm[0] for lm in landmarks]
        ys = [lm[1] for lm in landmarks]
        pad = 15  # padding pixels
        return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)

    @staticmethod
    def _get_finger_tips(landmarks: List[Tuple[int, int]]) -> Dict[str, Tuple[int, int]]:
        """Extract fingertip positions as a named dict."""
        return {name: landmarks[idx] for name, idx in _FINGERTIP_MAP.items()}

    @staticmethod
    def _get_hand_angle(landmarks: List[Tuple[int, int]]) -> float:
        """
        Compute hand rotation angle in degrees.
        Angle is measured from wrist (landmark 0) to middle MCP (landmark 9).
        0° = hand pointing up, positive = clockwise rotation.
        """
        wx, wy = landmarks[_WRIST]
        mx, my = landmarks[_MIDDLE_MCP]
        angle_rad = math.atan2(mx - wx, wy - my)  # Note: y is inverted in screen coords
        return math.degrees(angle_rad)
