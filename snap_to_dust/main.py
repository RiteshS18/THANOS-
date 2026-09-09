"""Run the Snap to Dust real-time webcam experience."""
from __future__ import annotations

import sys
import time
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

import config
from src.camera import Camera
from src.effects import add_dust_glow, apply_camera_shake, apply_flash
from src.hand_tracker import HandTracker
from src.particle_effect import ParticleEffect
from src.person_segmentation import PersonSegmenter
from src.snap_detector import SnapDetector
from src.utils import FpsMeter, clean_background, draw_text


class EffectState(str, Enum):
    NORMAL = "NORMAL"
    SNAP_DETECTED = "SNAP_DETECTED"
    DISINTEGRATING = "DISINTEGRATING"
    DISAPPEARED = "DISAPPEARED"
    RECONSTRUCTING = "RECONSTRUCTING"


def build_particle_effect(rng: np.random.Generator) -> ParticleEffect:
    return ParticleEffect(
        config.MAX_PARTICLES,
        config.PARTICLE_SAMPLE_STEP,
        rng,
        config.PARTICLE_MIN_LIFE,
        config.PARTICLE_MAX_LIFE,
        config.PARTICLE_MIN_SIZE,
        config.PARTICLE_MAX_SIZE,
        config.WIND_X,
        config.WIND_Y,
        config.GRAVITY,
        config.TURBULENCE,
        config.PARTICLE_DRAG,
    )


def draw_hud(frame: np.ndarray, state: EffectState, fps: float, particles: int, inference_ms: float, detector: SnapDetector, segmenter: PersonSegmenter, hand_tracker: HandTracker) -> None:
    draw_text(frame, f"FPS: {fps:4.1f}   STATE: {state.value}", (20, 32), (220, 245, 255), 0.62)
    draw_text(frame, f"Particles: {particles:,}   YOLO: {inference_ms:5.1f} ms", (20, 60), (190, 220, 235), 0.52)
    if config.DEBUG_MODE and state == EffectState.NORMAL:
        draw_text(frame, f"Snap: {detector.state.value}", (20, 88), (255, 218, 150), 0.5)
        draw_text(frame, f"Distance: {detector.distance:.3f}   Velocity: {detector.velocity:+.3f}", (20, 114), (255, 218, 150), 0.5)
    errors = [message for message in (segmenter.error, hand_tracker.error) if message]
    if errors:
        draw_text(frame, errors[-1][:110], (20, frame.shape[0] - 22), (120, 170, 255), 0.45)


def main() -> int:
    camera = Camera(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT)
    if not camera.open():
        print(f"Unable to open webcam index {config.CAMERA_INDEX}. Check permissions, camera index, and that no other app is using it.", file=sys.stderr)
        return 1

    print("Loading YOLO segmentation model and MediaPipe Hands...")
    segmenter = PersonSegmenter(config.MODEL_PATH, config.MODEL_FALLBACK, config.YOLO_IMGSZ, config.YOLO_CONFIDENCE, config.YOLO_DEVICE)
    hand_tracker = HandTracker(config.HAND_MODEL_PATH)
    detector = SnapDetector(
        config.SNAP_CLOSE_DISTANCE,
        config.SNAP_RELEASE_DISTANCE,
        config.SNAP_APPROACH_SPEED,
        config.SNAP_SEPARATION_SPEED,
        config.SNAP_COOLDOWN,
        config.SNAP_HISTORY_SIZE,
    )
    rng = np.random.default_rng(config.RNG_SEED)
    particles = build_particle_effect(rng)
    fps_meter = FpsMeter(config.FPS_SMOOTHING)
    state = EffectState.NORMAL
    state_started = time.perf_counter()
    last_frame_time = state_started
    frame_number = 0
    camera_failures = 0
    current_mask = None
    freeze_frame = None
    background = None
    flash_strength = 0.0
    camera_shake_frames = 0

    try:
        while True:
            frame = camera.read()
            if frame is None:
                camera_failures += 1
                if camera_failures < 30:
                    time.sleep(0.03)
                    continue
                print(f"Webcam stopped returning frames using {camera.backend_name or 'unknown'} backend.", file=sys.stderr)
                error_frame = np.zeros((480, 900, 3), dtype=np.uint8)
                draw_text(error_frame, "Camera opened but no frames arrived.", (30, 190), (100, 170, 255), 0.9)
                draw_text(error_frame, "Close Teams/Zoom/OBS, check Windows camera permissions, then press any key.", (30, 235), (225, 235, 245), 0.5)
                cv2.imshow(config.WINDOW_NAME, error_frame)
                cv2.waitKey(2500)
                break
            camera_failures = 0
            now = time.perf_counter()
            delta = now - last_frame_time
            last_frame_time = now
            fps = fps_meter.tick()
            frame_number += 1

            if state == EffectState.NORMAL:
                if frame_number % max(1, config.SEGMENTATION_INTERVAL) == 0 or current_mask is None:
                    current_mask = segmenter.detect(frame)
                landmarks = hand_tracker.process(frame)
                snap = detector.update(landmarks)
                display = frame.copy()
                if config.DEBUG_MODE:
                    hand_tracker.draw_landmarks(display)
                if config.SHOW_MASK and current_mask is not None:
                    display = segmenter.draw_mask(display, current_mask)
                if snap and current_mask is not None and int(current_mask.sum()) > 100:
                    if particles.start(frame, current_mask, config.DUST_DURATION):
                        freeze_frame = frame.copy()
                        background = clean_background(frame, current_mask)
                        state = EffectState.SNAP_DETECTED
                        state_started = now
                        flash_strength = 1.0
                        camera_shake_frames = 5
                        detector.reset()
                draw_hud(display, state, fps, particles.count, segmenter.last_inference_ms, detector, segmenter, hand_tracker)
            elif state == EffectState.SNAP_DETECTED:
                elapsed = now - state_started
                display = freeze_frame.copy() if freeze_frame is not None else frame.copy()
                if config.ENABLE_FLASH:
                    display = apply_flash(display, max(0.0, 1.0 - elapsed / config.FREEZE_DURATION) * flash_strength)
                if elapsed >= config.FREEZE_DURATION:
                    state = EffectState.DISINTEGRATING
                    state_started = now
                draw_hud(display, state, fps, particles.count, segmenter.last_inference_ms, detector, segmenter, hand_tracker)
            elif state == EffectState.DISINTEGRATING:
                finished = particles.update(delta)
                display = particles.render(background.copy() if background is not None else frame.copy())
                display = add_dust_glow(display, particles.particle_positions())
                if config.ENABLE_CAMERA_SHAKE and camera_shake_frames > 0:
                    display = apply_camera_shake(display, camera_shake_frames * 0.8, rng)
                    camera_shake_frames -= 1
                if finished or now - state_started >= config.DUST_DURATION:
                    state = EffectState.DISAPPEARED
                    state_started = now
                draw_hud(display, state, fps, particles.count, segmenter.last_inference_ms, detector, segmenter, hand_tracker)
            elif state == EffectState.DISAPPEARED:
                display = background.copy() if background is not None else frame.copy()
                if config.ENABLE_REAPPEAR and now - state_started >= config.REAPPEAR_DELAY:
                    state = EffectState.RECONSTRUCTING
                    state_started = now
                draw_hud(display, state, fps, particles.count, segmenter.last_inference_ms, detector, segmenter, hand_tracker)
            else:
                reconstruction_progress = min(1.0, (now - state_started) / max(config.RECONSTRUCT_DURATION, 0.1))
                particles.update(delta, reverse=True, reverse_progress=reconstruction_progress)
                display = particles.render(background.copy() if background is not None else frame.copy())
                display = add_dust_glow(display, particles.particle_positions(), 0.07)
                if now - state_started >= config.RECONSTRUCT_DURATION:
                    state = EffectState.NORMAL
                    current_mask = None
                    detector.reset()
                draw_hud(display, state, fps, particles.count, segmenter.last_inference_ms, detector, segmenter, hand_tracker)

            cv2.imshow(config.WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("r"):
                state = EffectState.NORMAL
                current_mask = None
                freeze_frame = None
                background = None
                detector.reset()
                particles = build_particle_effect(rng)
    finally:
        camera.release()
        hand_tracker.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
