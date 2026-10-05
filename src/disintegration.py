"""
disintegration.py — Thanos snap dust disintegration particle effect.

On snap trigger, captures the current frame, segments the person using
MediaPipe Image Segmenter (Tasks API), breaks the person region into a grid
of particles, and animates them drifting upward and fading away like ash.
"""

import math
import os
import random
from typing import Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np

# ─────────────────────────── Configurable Constants ───────────────────────────
PARTICLE_BLOCK_SIZE = 4          # Pixels — each particle represents this many px squared
EFFECT_DURATION_FRAMES = 180     # Total frames for the disintegration effect
DRIFT_SPEED_Y = -1.6             # Base upward drift (negative = up in screen coords)
DRIFT_SPEED_X = 0.35             # Base horizontal drift (slight rightward)
DRIFT_NOISE = 2.0                # Random noise added to velocity each frame
FADE_RATE = 0.008                # Alpha decrease per frame for a smoother vapor fade
SIZE_DECAY = 0.992               # Multiplicative size shrink per frame
HOLD_EMPTY_FRAMES = 120          # Frames to hold empty background after particles gone
WAVE_DELAY_MAX = 18              # Max delay frames for wave effect (left-to-right)
GRAVITY = 0.015                  # Slight downward pull (simulates air resistance turbulence)
SEGMENTATION_THRESHOLD = 0.6     # Person mask confidence threshold
BG_LEARNING_RATE = 0.005         # Rate at which background model updates
MAX_PARTICLES = 10000            # Cap on total particle count for performance

# Path to the selfie segmenter model file
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
SEGMENTER_MODEL_PATH = os.path.join(_PROJECT_DIR, "assets", "selfie_segmenter.tflite")
# ──────────────────────────────────────────────────────────────────────────────


class DisintegrationEffect:
    """
    Manages the Thanos-style dust disintegration effect.

    Workflow:
    1. Call `trigger(frame)` to start the effect — captures the person mask
       and generates particles.
    2. Call `update_and_render(live_frame)` each frame — returns the composited
       frame with particles over background.
    3. When `is_complete()` returns True, the effect is done.
    """

    def __init__(self, frame_width: int, frame_height: int):
        self.frame_w = frame_width
        self.frame_h = frame_height

        # MediaPipe Image Segmenter (Tasks API)
        self._segmentor = None
        self._segmentor_available = False
        self._init_segmenter()

        # Background model (running average)
        self._bg_model: Optional[np.ndarray] = None
        self._bg_ready = False

        # Effect state
        self._active = False
        self._frame_count = 0
        self._hold_count = 0

        # Particle data (NumPy arrays for vectorized updates)
        self._positions: Optional[np.ndarray] = None     # (N, 2) float32
        self._velocities: Optional[np.ndarray] = None    # (N, 2) float32
        self._colors: Optional[np.ndarray] = None        # (N, 3) uint8
        self._alphas: Optional[np.ndarray] = None        # (N,) float32
        self._sizes: Optional[np.ndarray] = None         # (N,) float32
        self._delays: Optional[np.ndarray] = None        # (N,) int32
        self._alive: Optional[np.ndarray] = None         # (N,) bool

        # Snapshot of frame at trigger time
        self._captured_frame: Optional[np.ndarray] = None
        self._background_frame: Optional[np.ndarray] = None
        self._person_mask: Optional[np.ndarray] = None

    def _init_segmenter(self):
        """Initialize the MediaPipe Image Segmenter if model is available."""
        if not os.path.exists(SEGMENTER_MODEL_PATH):
            print(f"  [Warning] Selfie segmenter model not found: {SEGMENTER_MODEL_PATH}")
            print("  Disintegration will use full-frame fallback.")
            self._segmentor_available = False
            return

        try:
            base_options = mp_python.BaseOptions(
                model_asset_path=SEGMENTER_MODEL_PATH
            )
            options = vision.ImageSegmenterOptions(
                base_options=base_options,
                output_category_mask=True,
                running_mode=vision.RunningMode.IMAGE,
            )
            self._segmentor = vision.ImageSegmenter.create_from_options(options)
            self._segmentor_available = True
        except Exception as e:
            print(f"  [Warning] Failed to init segmenter: {e}")
            self._segmentor_available = False

    def update_background(self, frame: np.ndarray):
        """
        Update the running average background model.
        Call this every frame when the effect is NOT active, so we have a
        clean background ready for compositing during disintegration.
        """
        if self._bg_model is None:
            self._bg_model = frame.astype(np.float32)
            self._bg_ready = True
        else:
            cv2.accumulateWeighted(frame, self._bg_model, BG_LEARNING_RATE)
            self._bg_ready = True

    def _build_fallback_mask(self, frame: np.ndarray) -> np.ndarray:
        """Build a foreground mask without relying on segmentation.

        This keeps the snapped subject isolated instead of turning the entire
        frame into a dissolve. The mask prefers the central figure and uses the
        background difference when available, giving a much more natural smoke
        and vapor effect.
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if self._bg_ready and self._bg_model is not None:
            diff = cv2.absdiff(frame, self._bg_model.astype(np.uint8))
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (31, 31), 0)
            _, thresh = cv2.threshold(
                blurred,
                max(15, int(np.mean(blurred) * 1.1)),
                255,
                cv2.THRESH_BINARY,
            )
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            cx, cy = w // 2, h // 2
            rx, ry = max(40, int(w * 0.22)), max(55, int(h * 0.30))
            body_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(body_mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
            mask = cv2.bitwise_and(thresh, body_mask)

        if mask.sum() < 2000:
            cx, cy = w // 2, h // 2
            rx, ry = max(40, int(w * 0.22)), max(55, int(h * 0.30))
            cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

        return mask.astype(np.uint8)

    def trigger(self, frame: np.ndarray):
        """
        Start the disintegration effect.

        Captures the current frame, runs person segmentation, and generates
        the particle system from the person-masked region.
        """
        self._captured_frame = frame.copy()
        self._active = True
        self._frame_count = 0
        self._hold_count = 0

        # Get person segmentation mask
        if self._segmentor_available and self._segmentor is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = self._segmentor.segment(mp_image)

                if results.category_mask is not None:
                    # category_mask is a MediaPipe Image; convert to numpy
                    mask_data = results.category_mask.numpy_view()
                    # Resize mask to frame size if needed
                    if mask_data.shape[:2] != (frame.shape[0], frame.shape[1]):
                        mask_data = cv2.resize(
                            mask_data.astype(np.float32),
                            (frame.shape[1], frame.shape[0]),
                            interpolation=cv2.INTER_LINEAR
                        )
                    self._person_mask = (mask_data > 0).astype(np.uint8)
                elif results.confidence_masks and len(results.confidence_masks) > 0:
                    # Use confidence mask as fallback
                    mask_data = results.confidence_masks[0].numpy_view()
                    if mask_data.shape[:2] != (frame.shape[0], frame.shape[1]):
                        mask_data = cv2.resize(
                            mask_data,
                            (frame.shape[1], frame.shape[0]),
                            interpolation=cv2.INTER_LINEAR
                        )
                    self._person_mask = (mask_data > SEGMENTATION_THRESHOLD).astype(np.uint8)
                else:
                    self._person_mask = self._build_fallback_mask(frame)
            except Exception as e:
                print(f"  [Warning] Segmentation failed: {e}")
                self._person_mask = self._build_fallback_mask(frame)
        else:
            self._person_mask = self._build_fallback_mask(frame)

        # Use background model or a blurred version of the current frame
        if self._bg_ready and self._bg_model is not None:
            self._background_frame = self._bg_model.astype(np.uint8)
        else:
            # Fallback: heavily blur the current frame as pseudo-background
            self._background_frame = cv2.GaussianBlur(frame, (51, 51), 0)

        # Generate particles from the person mask
        self._generate_particles(frame)

    def update_and_render(self, live_frame: np.ndarray) -> np.ndarray:
        """
        Update particle positions and render the effect.

        Args:
            live_frame: Current live webcam frame (used as fallback background).

        Returns:
            Composited frame showing the disintegration effect.
        """
        if not self._active:
            return live_frame

        self._frame_count += 1

        # Start with background
        if self._background_frame is not None:
            output = self._background_frame.copy()
        else:
            output = live_frame.copy()

        # Check if we still have live particles
        if self._positions is not None and self._alive is not None:
            alive_count = np.sum(self._alive)

            if alive_count > 0:
                self._update_particles()
                self._render_particles(output)
            else:
                # All particles dead — enter hold phase
                self._hold_count += 1
        else:
            self._hold_count += 1

        # Check duration limits
        if self._frame_count > EFFECT_DURATION_FRAMES:
            self._hold_count = max(self._hold_count, HOLD_EMPTY_FRAMES - 30)

        # Draw effect progress indicator
        self._draw_snap_overlay(output)

        return output

    def is_active(self) -> bool:
        """Check if the effect is currently running."""
        return self._active

    def is_complete(self) -> bool:
        """Check if the effect has fully completed (particles done + hold done)."""
        return self._active and self._hold_count >= HOLD_EMPTY_FRAMES

    def reset(self):
        """Reset the effect state for a new round."""
        self._active = False
        self._frame_count = 0
        self._hold_count = 0
        self._positions = None
        self._velocities = None
        self._colors = None
        self._alphas = None
        self._sizes = None
        self._delays = None
        self._alive = None
        self._captured_frame = None
        self._person_mask = None

    def release(self):
        """Release MediaPipe resources."""
        if self._segmentor is not None:
            self._segmentor.close()

    # ──────────────────────── Private Methods ────────────────────────

    def _generate_particles(self, frame: np.ndarray):
        """
        Generate particles from the person-masked region of the frame.
        Each PARTICLE_BLOCK_SIZE × PARTICLE_BLOCK_SIZE block becomes one particle.
        """
        h, w = frame.shape[:2]
        mask = self._person_mask

        positions = []
        colors = []

        bs = PARTICLE_BLOCK_SIZE
        for y in range(0, h - bs, bs):
            for x in range(0, w - bs, bs):
                # Check if this block is inside the person mask
                block_mask = mask[y:y + bs, x:x + bs]
                if np.mean(block_mask) < 0.5:
                    continue

                # Sample the average color of this block
                block_color = frame[y:y + bs, x:x + bs]
                avg_color = block_color.mean(axis=(0, 1)).astype(np.uint8)

                positions.append([float(x + bs // 2), float(y + bs // 2)])
                colors.append(avg_color)

                if len(positions) >= MAX_PARTICLES:
                    break
            if len(positions) >= MAX_PARTICLES:
                break

        n = len(positions)
        if n == 0:
            self._alive = np.zeros(0, dtype=bool)
            return

        self._positions = np.array(positions, dtype=np.float32)        # (N, 2)
        self._colors = np.array(colors, dtype=np.uint8)                # (N, 3)

        # Randomized velocities: upward drift + horizontal noise
        self._velocities = np.column_stack([
            np.random.uniform(-1.0, 1.6, n).astype(np.float32),
            np.random.uniform(-3.6, -1.0, n).astype(np.float32),
        ]).astype(np.float32)

        self._alphas = np.ones(n, dtype=np.float32)
        self._sizes = np.full(n, float(bs * 1.4), dtype=np.float32)
        self._alive = np.ones(n, dtype=bool)

        # Wave delay: particles on the left start earlier, right side later
        # This creates the iconic left-to-right dissolve
        x_positions = self._positions[:, 0]
        x_norm = (x_positions - x_positions.min()) / max(1, x_positions.max() - x_positions.min())
        self._delays = (x_norm * WAVE_DELAY_MAX).astype(np.int32)

        # Add some randomness to delays
        self._delays += np.random.randint(0, 8, n).astype(np.int32)

    def _update_particles(self):
        """Update particle positions, alphas, and sizes (vectorized)."""
        # Only update particles past their delay
        active = self._alive & (self._frame_count >= self._delays)

        if not np.any(active):
            return

        # Add turbulence noise to velocities
        n_active = np.sum(active)
        noise = np.random.normal(0, DRIFT_NOISE, (n_active, 2)).astype(np.float32)
        self._velocities[active] += noise

        # Apply slight gravity (downward pull to create arc)
        self._velocities[active, 1] += GRAVITY

        # Update positions
        self._positions[active] += self._velocities[active]

        # Fade alpha more softly so the body turns into a vapor cloud
        self._alphas[active] -= FADE_RATE * (0.8 + np.random.random(n_active) * 0.7)

        # Shrink size
        self._sizes[active] *= SIZE_DECAY

        # Kill particles that are fully faded, too small, or off-screen
        off_screen = (
            (self._positions[:, 0] < -50) |
            (self._positions[:, 0] > self.frame_w + 50) |
            (self._positions[:, 1] < -50) |
            (self._positions[:, 1] > self.frame_h + 50)
        )
        too_faded = self._alphas <= 0
        too_small = self._sizes < 0.5

        self._alive &= ~(off_screen | too_faded | too_small)

    def _render_particles(self, output: np.ndarray):
        """Render alive particles onto the output frame."""
        # For particles still in their delay period, draw them from the captured frame
        # (they haven't started moving yet, so show original person pixels)
        if self._captured_frame is not None and self._person_mask is not None:
            delayed = self._alive & (self._frame_count < self._delays)
            if np.any(delayed):
                delayed_indices = np.where(delayed)[0]
                for idx in delayed_indices:
                    x, y = int(self._positions[idx, 0]), int(self._positions[idx, 1])
                    sz = int(self._sizes[idx])
                    half = sz // 2
                    if 0 <= y - half < output.shape[0] and 0 <= x - half < output.shape[1]:
                        y1 = max(0, y - half)
                        y2 = min(output.shape[0], y + half)
                        x1 = max(0, x - half)
                        x2 = min(output.shape[1], x + half)
                        if y2 > y1 and x2 > x1:
                            src_y1 = max(0, y - half)
                            src_y2 = min(self._captured_frame.shape[0], y + half)
                            src_x1 = max(0, x - half)
                            src_x2 = min(self._captured_frame.shape[1], x + half)
                            h_out = min(y2 - y1, src_y2 - src_y1)
                            w_out = min(x2 - x1, src_x2 - src_x1)
                            if h_out > 0 and w_out > 0:
                                output[y1:y1+h_out, x1:x1+w_out] = \
                                    self._captured_frame[src_y1:src_y1+h_out, src_x1:src_x1+w_out]

        # Draw moving particles
        active = self._alive & (self._frame_count >= self._delays)
        if not np.any(active):
            return

        # Create a particle overlay layer
        overlay = output.copy()

        active_indices = np.where(active)[0]

        for idx in active_indices:
            x = int(self._positions[idx, 0])
            y = int(self._positions[idx, 1])
            sz = max(1, int(self._sizes[idx]))
            alpha = float(self._alphas[idx])
            base_color = np.array(self._colors[idx], dtype=np.float32)
            soft_color = np.clip(base_color * (0.8 + alpha * 0.6), 0, 255).astype(np.uint8)

            if alpha <= 0 or sz < 1:
                continue

            # Draw a soft smoke-like dust particle.
            radius = max(1, sz // 2)
            cv2.circle(overlay, (x, y), radius, tuple(int(v) for v in soft_color), -1)
            if radius > 1:
                cv2.circle(overlay, (x, y), radius + 1,
                           tuple(int(v * 0.35) for v in soft_color), 1)

        # Blend overlay with alpha so the vapor hangs in the air instead of
        # vanishing immediately.
        avg_alpha = float(np.mean(self._alphas[active_indices]))
        blend_alpha = max(0.18, min(0.9, 0.25 + avg_alpha * 0.75))
        cv2.addWeighted(overlay, blend_alpha, output, 1 - blend_alpha, 0, output)

    def _draw_snap_overlay(self, output: np.ndarray):
        """Draw the 'THE SNAP' text and flash effect."""
        h, w = output.shape[:2]

        # Initial flash effect (first few frames)
        if self._frame_count < 8:
            flash_alpha = max(0, 1.0 - self._frame_count / 8.0)
            flash = np.full_like(output, 255)
            cv2.addWeighted(flash, flash_alpha * 0.7, output, 1.0, 0, output)

        # "THE SNAP" text with glow
        if self._frame_count < EFFECT_DURATION_FRAMES:
            text = "THE SNAP"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1.8
            thickness = 3
            text_size = cv2.getTextSize(text, font, scale, thickness)[0]
            tx = (w - text_size[0]) // 2
            ty = h // 2

            # Only show text prominently in the first portion
            text_alpha = max(0, 1.0 - self._frame_count / 60.0)
            if text_alpha > 0.05:
                # Glow behind text
                for offset in [(0, 0), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                    cv2.putText(output, text,
                                (tx + offset[0] * 2, ty + offset[1] * 2),
                                font, scale, (100, 50, 200),
                                thickness + 2, cv2.LINE_AA)
                # Main text
                cv2.putText(output, text, (tx, ty), font, scale,
                            (255, 255, 255), thickness, cv2.LINE_AA)
