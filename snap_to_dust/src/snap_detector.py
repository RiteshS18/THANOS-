"""Temporal finger-snap detector based on MediaPipe landmarks."""
from __future__ import annotations

import time
from collections import deque
from enum import Enum
from math import hypot


class SnapState(str, Enum):
    IDLE = "IDLE"
    FINGERS_APPROACH = "FINGERS_APPROACH"
    CONTACT = "CONTACT/CLOSE"
    RAPID_SEPARATION = "RAPID_SEPARATION"
    COOLDOWN = "COOLDOWN"


class SnapDetector:
    THUMB_TIP = 4
    INDEX_TIP = 8
    INDEX_MCP = 5
    THUMB_MCP = 2
    WRIST = 0

    def __init__(self, close_distance: float, release_distance: float, approach_speed: float, separation_speed: float, cooldown: float, history_size: int = 12) -> None:
        self.close_distance = close_distance
        self.release_distance = release_distance
        self.approach_speed = approach_speed
        self.separation_speed = separation_speed
        self.cooldown = cooldown
        self.state = SnapState.IDLE
        self.last_event = 0.0
        self.previous_distance: float | None = None
        self.distance_history: deque[float] = deque(maxlen=history_size)
        self.distance = 0.0
        self.velocity = 0.0

    def _distance(self, landmarks) -> float | None:
        if landmarks is None or len(landmarks) <= self.INDEX_TIP:
            return None
        thumb = landmarks[self.THUMB_TIP]
        index = landmarks[self.INDEX_TIP]
        wrist = landmarks[self.WRIST]
        index_mcp = landmarks[self.INDEX_MCP]
        scale = max(hypot(index_mcp[0] - wrist[0], index_mcp[1] - wrist[1]), 1e-4)
        return hypot(thumb[0] - index[0], thumb[1] - index[1]) / scale

    def update(self, landmarks, now: float | None = None) -> bool:
        current_time = time.monotonic() if now is None else now
        distance = self._distance(landmarks)
        if distance is None:
            if self.state != SnapState.COOLDOWN:
                self.state = SnapState.IDLE
            self.velocity = 0.0
            return False
        prior_distance = self.previous_distance
        self.distance = distance
        self.distance_history.append(distance)
        self.velocity = 0.0 if prior_distance is None else distance - prior_distance
        self.previous_distance = distance
        if self.state == SnapState.COOLDOWN:
            if current_time - self.last_event >= self.cooldown:
                self.state = SnapState.IDLE
            return False
        if self.state == SnapState.IDLE:
            if distance <= self.release_distance:
                self.state = SnapState.FINGERS_APPROACH
        elif self.state == SnapState.FINGERS_APPROACH:
            if distance <= self.close_distance:
                self.state = SnapState.CONTACT
            elif prior_distance is not None and prior_distance <= self.close_distance * 1.5 and distance >= self.release_distance and self.velocity >= self.separation_speed:
                self.last_event = current_time
                self.state = SnapState.COOLDOWN
                return True
            elif self.velocity > 0.03:
                self.state = SnapState.IDLE
        elif self.state == SnapState.CONTACT:
            if distance >= self.release_distance and self.velocity >= self.separation_speed:
                self.last_event = current_time
                self.state = SnapState.COOLDOWN
                return True
        elif self.state == SnapState.RAPID_SEPARATION:
            self.last_event = current_time
            self.state = SnapState.COOLDOWN
            return True
        return False

    def reset(self) -> None:
        self.state = SnapState.IDLE
        self.previous_distance = None
        self.distance_history.clear()
        self.velocity = 0.0
