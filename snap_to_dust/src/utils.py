"""Small image and timing helpers."""
from __future__ import annotations

import time
from typing import Iterable

import cv2
import numpy as np


def draw_text(frame: np.ndarray, text: str, origin: tuple[int, int], color=(235, 245, 255), scale=0.58) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (15, 18, 24), 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def clean_background(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill the person area using nearby pixels so the dust has a clean backdrop."""
    mask_u8 = (mask.astype(np.uint8) * 255) if mask.dtype != np.uint8 else mask
    expanded = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    expanded = cv2.dilate(expanded, np.ones((5, 5), np.uint8), iterations=1)
    try:
        return cv2.inpaint(frame, expanded, 3, cv2.INPAINT_TELEA)
    except cv2.error:
        return frame.copy()


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0


class FpsMeter:
    def __init__(self, smoothing: float = 0.9) -> None:
        self.smoothing = smoothing
        self.last_time = time.perf_counter()
        self.value = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        instant = 1.0 / max(now - self.last_time, 1e-6)
        self.last_time = now
        self.value = instant if self.value == 0 else self.value * self.smoothing + instant * (1 - self.smoothing)
        return self.value


def finite_or_zero(values: Iterable[float]) -> list[float]:
    return [float(value) if np.isfinite(value) else 0.0 for value in values]
