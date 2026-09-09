"""Lightweight cinematic overlays."""
from __future__ import annotations

import cv2
import numpy as np


def apply_flash(frame: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return frame
    white = np.full_like(frame, 255)
    return cv2.addWeighted(frame, 1.0 - strength, white, strength, 0)


def apply_camera_shake(frame: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    if strength <= 0:
        return frame
    height, width = frame.shape[:2]
    offset_x = int(rng.uniform(-strength, strength))
    offset_y = int(rng.uniform(-strength, strength))
    matrix = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    return cv2.warpAffine(frame, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)


def add_dust_glow(frame: np.ndarray, positions: np.ndarray, alpha: float = 0.12) -> np.ndarray:
    if len(positions) == 0 or alpha <= 0:
        return frame
    cloud = np.zeros_like(frame)
    height, width = frame.shape[:2]
    for x, y in positions[:: max(1, len(positions) // 250)]:
        ix, iy = int(x), int(y)
        if 0 <= ix < width and 0 <= iy < height:
            cv2.circle(cloud, (ix, iy), 5, (180, 205, 220), -1, cv2.LINE_AA)
    cloud = cv2.GaussianBlur(cloud, (0, 0), 8)
    return cv2.addWeighted(frame, 1.0, cloud, alpha, 0)
