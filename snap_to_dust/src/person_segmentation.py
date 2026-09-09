"""YOLO person segmentation with a CPU-safe fallback."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np


class PersonSegmenter:
    def __init__(self, model_path: str, fallback_model: str, image_size: int, confidence: float, device: str = "") -> None:
        self.model = None
        self.image_size = image_size
        self.confidence = confidence
        self.device = device or None
        self.last_inference_ms = 0.0
        self.error: str | None = None
        try:
            from ultralytics import YOLO

            selected_model = model_path if Path(model_path).exists() else fallback_model
            self.model = YOLO(selected_model)
        except Exception as exc:
            self.error = f"YOLO unavailable: {exc}"

    @property
    def ready(self) -> bool:
        return self.model is not None

    def get_person_mask(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        empty = np.zeros((height, width), dtype=bool)
        if self.model is None:
            return empty
        started = time.perf_counter()
        try:
            results = self.model.predict(
                source=frame,
                imgsz=self.image_size,
                conf=self.confidence,
                classes=[0],
                device=self.device,
                verbose=False,
            )
            self.last_inference_ms = (time.perf_counter() - started) * 1000.0
            result = results[0]
            if result.masks is None or result.boxes is None or len(result.masks.data) == 0:
                return empty
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            confidences = result.boxes.conf.detach().cpu().numpy()
            masks = result.masks.data.detach().cpu().numpy()
            best_index = -1
            best_area = 0.0
            for index, (class_id, confidence) in enumerate(zip(classes, confidences)):
                if class_id != 0 or confidence < self.confidence:
                    continue
                area = float(masks[index].sum())
                if area > best_area:
                    best_index, best_area = index, area
            if best_index < 0:
                return empty
            mask = cv2.resize(masks[best_index], (width, height), interpolation=cv2.INTER_LINEAR)
            mask = (mask > 0.45).astype(np.uint8) * 255
            kernel_size = 5
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            return mask > 100
        except Exception as exc:
            self.error = f"Segmentation failed: {exc}"
            return empty

    def detect(self, frame: np.ndarray) -> np.ndarray:
        return self.get_person_mask(frame)

    def draw_mask(self, frame: np.ndarray, mask: np.ndarray, color=(40, 185, 255), alpha=0.24) -> np.ndarray:
        overlay = frame.copy()
        overlay[mask] = color
        return cv2.addWeighted(frame, 1.0 - alpha, overlay, alpha, 0)
