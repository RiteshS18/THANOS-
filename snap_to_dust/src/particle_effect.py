"""Vectorized dust particle simulation and rendering."""
from __future__ import annotations

import cv2
import numpy as np


class ParticleEffect:
    def __init__(self, max_particles: int, sample_step: int, rng: np.random.Generator, min_life: float, max_life: float, min_size: float, max_size: float, wind_x: float, wind_y: float, gravity: float, turbulence: float, drag: float) -> None:
        self.max_particles = max_particles
        self.sample_step = max(1, sample_step)
        self.rng = rng
        self.min_life, self.max_life = min_life, max_life
        self.min_size, self.max_size = min_size, max_size
        self.wind = np.array([wind_x, wind_y], dtype=np.float32)
        self.gravity = gravity
        self.turbulence = turbulence
        self.drag = drag
        self.count = 0
        self.x = self.y = self.original_x = self.original_y = np.empty(0, dtype=np.float32)
        self.vx = self.vy = self.size = self.opacity = self.life = self.age = np.empty(0, dtype=np.float32)
        self.colors = np.empty((0, 3), dtype=np.uint8)
        self.duration = 1.0

    def start(self, frame: np.ndarray, mask: np.ndarray, duration: float) -> bool:
        ys, xs = np.where(mask[:: self.sample_step, :: self.sample_step])
        xs = xs * self.sample_step
        ys = ys * self.sample_step
        if len(xs) == 0:
            self.count = 0
            return False
        if len(xs) > self.max_particles:
            selected = self.rng.choice(len(xs), self.max_particles, replace=False)
            xs, ys = xs[selected], ys[selected]
        self.count = len(xs)
        self.original_x = xs.astype(np.float32)
        self.original_y = ys.astype(np.float32)
        self.x = self.original_x.copy()
        self.y = self.original_y.copy()
        self.vx = self.rng.normal(0.0, 0.5, self.count).astype(np.float32)
        self.vy = self.rng.normal(-0.65, 0.35, self.count).astype(np.float32)
        self.size = self.rng.uniform(self.min_size, self.max_size, self.count).astype(np.float32)
        self.opacity = np.ones(self.count, dtype=np.float32)
        self.life = self.rng.uniform(self.min_life, self.max_life, self.count).astype(np.float32)
        self.age = np.zeros(self.count, dtype=np.float32)
        self.colors = frame[ys, xs].copy()
        self.duration = max(duration, 0.1)
        return True

    def update(self, delta_seconds: float, reverse: bool = False, reverse_progress: float = 0.0) -> bool:
        if self.count == 0:
            return True
        dt = min(max(delta_seconds, 0.0), 0.08)
        if reverse:
            progress = min(1.0, float(self.age.mean() / self.duration) if self.count else 1.0)
            stiffness = 0.018 + progress * 0.035
            self.vx += (self.original_x - self.x) * stiffness * dt * 60.0
            self.vy += (self.original_y - self.y) * stiffness * dt * 60.0
            self.age = np.maximum(0.0, self.age - dt)
            fade = np.clip(reverse_progress / 0.35, 0.0, 1.0)
            self.opacity = np.full(self.count, np.clip((0.12 + progress * 0.88) * fade, 0.0, 1.0), dtype=np.float32)
            self.x += self.vx * dt * 60.0
            self.y += self.vy * dt * 60.0
            self.vx *= 0.91
            self.vy *= 0.91
            return bool(np.mean(np.abs(self.x - self.original_x) + np.abs(self.y - self.original_y)) < 1.8)
        self.age += dt
        turbulence = self.rng.normal(0.0, self.turbulence, (self.count, 2)).astype(np.float32)
        self.vx += self.wind[0] * dt + turbulence[:, 0] * dt * 8.0
        self.vy += (self.wind[1] + self.gravity) * dt + turbulence[:, 1] * dt * 8.0
        self.vx *= self.drag
        self.vy *= self.drag
        self.x += self.vx * dt * 60.0
        self.y += self.vy * dt * 60.0
        self.opacity = np.clip(1.0 - self.age / self.life, 0.0, 1.0)
        return bool(float(np.mean(self.opacity)) <= 0.015)

    def render(self, frame: np.ndarray) -> np.ndarray:
        if self.count == 0:
            return frame
        output = frame.copy()
        height, width = output.shape[:2]
        visible = (self.opacity > 0.01) & (self.x >= 0) & (self.x < width) & (self.y >= 0) & (self.y < height)
        particle_layer = np.zeros_like(output)
        alpha_layer = np.zeros((height, width), dtype=np.uint8)
        for index in np.flatnonzero(visible):
            radius = max(1, int(self.size[index] * (0.6 + self.opacity[index])))
            color = tuple(int(channel) for channel in self.colors[index])
            alpha = int(np.clip(self.opacity[index] * 220.0, 0.0, 220.0))
            center = (int(self.x[index]), int(self.y[index]))
            cv2.circle(particle_layer, center, radius, color, -1, cv2.LINE_AA)
            cv2.circle(alpha_layer, center, radius, alpha, -1, cv2.LINE_AA)
        alpha_values = alpha_layer.astype(np.float32)[:, :, None] / 255.0
        return (output.astype(np.float32) * (1.0 - alpha_values) + particle_layer.astype(np.float32) * alpha_values).astype(np.uint8)

    def particle_positions(self) -> np.ndarray:
        if self.count == 0:
            return np.empty((0, 2), dtype=np.float32)
        return np.column_stack((self.x, self.y))
