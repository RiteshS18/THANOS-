"""
main.py — Infinity Gauntlet: The Snap — Main Application Loop

Ties together hand tracking, gauntlet overlay, stone collection, snap detection,
and the dust disintegration effect into a single real-time pipeline.

Controls:
    - Move your hand near floating stones to collect them
    - Once all 6 stones are collected, pinch thumb + middle finger to SNAP
    - Press 'Q' to quit
    - Press 'R' to reset
    - Press 'D' to toggle debug overlay

Usage:
    python -m src.main
    OR
    python src/main.py
"""

import math
import sys
import os
import time
from typing import Optional

import cv2
import numpy as np

# Add parent dir to path so we can import src modules when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hand_tracker import HandTracker
from src.gauntlet_overlay import GauntletOverlay, STONE_COLORS
from src.stone_manager import StoneManager
from src.snap_detector import SnapDetector
from src.disintegration import DisintegrationEffect

# ─────────────────────────── Configurable Constants ───────────────────────────
CAMERA_INDEX = 0                 # Webcam device index
FRAME_WIDTH = 640                # Capture width
FRAME_HEIGHT = 480               # Capture height
WINDOW_NAME = "Infinity Gauntlet: The Snap"
TARGET_FPS = 30                  # Target frame rate (for timing)
SHOW_DEBUG = False               # Start with debug overlay off
# ──────────────────────────────────────────────────────────────────────────────

# Application states
STATE_COLLECTING = "COLLECTING"
STATE_READY_TO_SNAP = "READY_TO_SNAP"
STATE_DISINTEGRATING = "DISINTEGRATING"
STATE_RESET_HOLD = "RESET_HOLD"

# HUD styling
HUD_BG_ALPHA = 0.65
HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_FONT_SMALL = cv2.FONT_HERSHEY_PLAIN
HUD_COLOR_TEXT = (240, 240, 240)
HUD_COLOR_DIM = (150, 150, 150)
HUD_COLOR_ACCENT = (100, 220, 255)     # Gold accent
HUD_COLOR_READY = (50, 255, 50)        # Green ready
HUD_COLOR_SNAP = (180, 80, 255)        # Purple snap

# Optional audio
try:
    import pygame
    pygame.mixer.init()
    AUDIO_AVAILABLE = True
except (ImportError, Exception):
    AUDIO_AVAILABLE = False


def load_sfx(filename: str) -> Optional[object]:
    """Try to load a sound effect file from assets/sfx/."""
    if not AUDIO_AVAILABLE:
        return None
    sfx_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "assets", "sfx", filename
    )
    if os.path.exists(sfx_path):
        try:
            return pygame.mixer.Sound(sfx_path)
        except Exception:
            pass
    return None


class InfinityGauntletApp:
    """
    Main application class for the Infinity Gauntlet: The Snap experience.

    Manages the state machine, frame pipeline, and HUD rendering.
    """

    def __init__(self):
        # Initialize webcam
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open webcam at index {CAMERA_INDEX}. "
                "Check your camera connection."
            )

        # Get actual frame dimensions (may differ from requested)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize components
        self.tracker = HandTracker()
        self.gauntlet = GauntletOverlay()
        self.stone_mgr = StoneManager(self.w, self.h)
        self.snap_det = SnapDetector()
        self.disintegration = DisintegrationEffect(self.w, self.h)

        # Application state
        self.state = STATE_COLLECTING
        self.debug_mode = SHOW_DEBUG
        self.frame_count = 0
        self.fps = 0.0
        self._fps_timer = time.time()
        self._fps_counter = 0
        self._reset_hold_counter = 0

        # Audio
        self.sfx_collect = load_sfx("collect.wav")
        self.sfx_snap = load_sfx("snap.wav")

        # Collection flash effect
        self._flash_frames = 0
        self._flash_color = (255, 255, 255)

    def run(self):
        """Main application loop."""
        print(f"\n{'='*55}")
        print("  INFINITY GAUNTLET: THE SNAP")
        print(f"{'='*55}")
        print(f"  Camera: {self.w}x{self.h} @ index {CAMERA_INDEX}")
        print(f"  Audio:  {'Enabled' if AUDIO_AVAILABLE else 'Disabled'}")
        print(f"{'─'*55}")
        print("  Controls:")
        print("    • Move hand to stones to collect them")
        print("    • Pinch thumb + middle finger to SNAP")
        print("    • [D] Toggle debug overlay")
        print("    • [R] Reset")
        print("    • [Q] Quit")
        print(f"{'='*55}\n")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Cannot read from webcam.")
                    break

                # Mirror the frame for natural interaction
                frame = cv2.flip(frame, 1)

                # Process frame through the pipeline
                output = self._process_frame(frame)

                # Draw HUD
                output = self._draw_hud(output)

                # Update FPS counter
                self._update_fps()

                # Display
                cv2.imshow(WINDOW_NAME, output)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('d') or key == ord('D'):
                    self.debug_mode = not self.debug_mode
                elif key == ord('r') or key == ord('R'):
                    self._reset_all()

                self.frame_count += 1

        except KeyboardInterrupt:
            pass
        finally:
            self._cleanup()

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame through the entire pipeline based on current state.
        """
        # ── STATE: DISINTEGRATING ──
        if self.state == STATE_DISINTEGRATING:
            output = self.disintegration.update_and_render(frame)
            if self.disintegration.is_complete():
                self.state = STATE_RESET_HOLD
                self._reset_hold_counter = 0
            return output

        # ── STATE: RESET_HOLD ──
        if self.state == STATE_RESET_HOLD:
            self._reset_hold_counter += 1
            if self._reset_hold_counter > 60:  # ~2 seconds
                self._reset_all()
            return frame

        # ── STATES: COLLECTING / READY_TO_SNAP ──

        # Step 1: Hand tracking
        hands = self.tracker.detect(frame)

        # Step 2: Update background model for disintegration
        self.disintegration.update_background(frame)

        output = frame.copy()

        if hands:
            hand = hands[0]  # Use first detected hand

            # Step 3: Gauntlet overlay
            collected_state = self.stone_mgr.get_collection_state()
            output = self.gauntlet.render(
                output, hand.landmarks, collected_state, hand.hand_angle
            )

            # Step 4: Stone collection (only in COLLECTING state)
            if self.state == STATE_COLLECTING:
                just_collected = self.stone_mgr.update(
                    hand.palm_center, hand.finger_tips, hand.landmarks
                )
                if just_collected:
                    self._on_stone_collected(just_collected)

                # Check if all stones collected
                if self.stone_mgr.all_collected():
                    self.state = STATE_READY_TO_SNAP
                    self.snap_det.arm()

            # Step 5: Snap detection (only in READY_TO_SNAP state)
            if self.state == STATE_READY_TO_SNAP:
                snap_triggered = self.snap_det.update(hand.finger_tips)
                if snap_triggered:
                    self._on_snap(output)

            # Debug overlay
            if self.debug_mode:
                output = self.tracker.draw_debug(output, hands)

        else:
            # No hand detected
            if self.state == STATE_COLLECTING:
                self.stone_mgr.update(None, None)
            self.snap_det.update(None)

        # Step 6: Render floating stones
        output = self.stone_mgr.render(output)

        # Apply collection flash
        if self._flash_frames > 0:
            self._flash_frames -= 1
            flash_alpha = self._flash_frames / 12.0
            flash_overlay = np.full_like(output, self._flash_color, dtype=np.uint8)
            cv2.addWeighted(flash_overlay, flash_alpha * 0.25, output, 1.0, 0, output)

        return output

    def _on_stone_collected(self, stone_name: str):
        """Handle stone collection event."""
        color = STONE_COLORS.get(stone_name, (255, 255, 255))
        self._flash_color = color
        self._flash_frames = 12
        if self.sfx_collect:
            try:
                self.sfx_collect.play()
            except Exception:
                pass
        count = self.stone_mgr.collected_count()
        print(f"  ✦ Collected: {stone_name.upper()} stone ({count}/6)")

    def _on_snap(self, frame: np.ndarray):
        """Handle snap trigger event."""
        self.state = STATE_DISINTEGRATING
        self.disintegration.trigger(frame)
        if self.sfx_snap:
            try:
                self.sfx_snap.play()
            except Exception:
                pass
        print("\n  ⚡ SNAP! ⚡\n")

    def _reset_all(self):
        """Reset the entire application state."""
        self.state = STATE_COLLECTING
        self.stone_mgr.reset()
        self.snap_det.reset()
        self.disintegration.reset()
        self._flash_frames = 0
        print("  ↺ Reset — collect the stones again!")

    def _draw_hud(self, frame: np.ndarray) -> np.ndarray:
        """Draw the heads-up display overlay."""
        h, w = frame.shape[:2]
        output = frame.copy()

        # ── Bottom HUD bar ──
        bar_height = 70
        bar_y = h - bar_height

        # Semi-transparent dark bar
        bar_overlay = output.copy()
        cv2.rectangle(bar_overlay, (0, bar_y), (w, h), (20, 20, 25), -1)
        cv2.addWeighted(bar_overlay, HUD_BG_ALPHA, output, 1 - HUD_BG_ALPHA, 0, output)

        # Thin accent line at top of bar
        cv2.line(output, (0, bar_y), (w, bar_y), (80, 80, 90), 1)

        # ── Stone indicator dots ──
        dot_y = bar_y + 22
        dot_start_x = 20
        dot_spacing = 38
        collected_state = self.stone_mgr.get_collection_state()
        stone_order = ["power", "space", "reality", "soul", "mind", "time"]

        for i, name in enumerate(stone_order):
            cx = dot_start_x + i * dot_spacing
            color = STONE_COLORS.get(name, (200, 200, 200))
            is_collected = collected_state.get(name, False)

            if is_collected:
                # Filled bright dot with glow
                cv2.circle(output, (cx, dot_y), 10, color, -1, cv2.LINE_AA)
                cv2.circle(output, (cx, dot_y), 13, color, 1, cv2.LINE_AA)
                # Specular
                cv2.circle(output, (cx - 3, dot_y - 3), 3,
                           (255, 255, 255), -1, cv2.LINE_AA)
            else:
                # Outlined dim dot
                dim_color = tuple(c // 3 for c in color)
                cv2.circle(output, (cx, dot_y), 10, dim_color, 2, cv2.LINE_AA)

        # ── Status text ──
        count = self.stone_mgr.collected_count()
        status_x = dot_start_x + len(stone_order) * dot_spacing + 15

        if self.state == STATE_COLLECTING:
            status_text = f"Collect the Stones  ({count}/6)"
            status_color = HUD_COLOR_TEXT
        elif self.state == STATE_READY_TO_SNAP:
            status_text = "ALL STONES COLLECTED  -  SNAP NOW!"
            status_color = HUD_COLOR_READY
            # Pulsing effect
            pulse = int(80 * abs(math.sin(self.frame_count * 0.1)))
            status_color = (50 + pulse, 255, 50 + pulse)
        elif self.state == STATE_DISINTEGRATING:
            status_text = "THE SNAP"
            status_color = HUD_COLOR_SNAP
        else:
            status_text = "Resetting..."
            status_color = HUD_COLOR_DIM

        cv2.putText(output, status_text, (status_x, dot_y + 5),
                    HUD_FONT, 0.55, status_color, 1, cv2.LINE_AA)

        # ── Instructions text ──
        if self.state == STATE_COLLECTING:
            instr = "Move hand to stones | [D] Debug | [R] Reset | [Q] Quit"
        elif self.state == STATE_READY_TO_SNAP:
            instr = "Pinch THUMB + MIDDLE finger to snap! | [R] Reset"
        elif self.state == STATE_DISINTEGRATING:
            instr = "..."
        else:
            instr = "Resetting..."

        cv2.putText(output, instr, (20, bar_y + 55),
                    HUD_FONT_SMALL, 1.0, HUD_COLOR_DIM, 1, cv2.LINE_AA)

        # ── FPS counter (top-left) ──
        fps_text = f"FPS: {self.fps:.0f}"
        cv2.putText(output, fps_text, (12, 22),
                    HUD_FONT_SMALL, 1.0, (100, 100, 100), 1, cv2.LINE_AA)

        # ── Title (top-center) ──
        title = "INFINITY GAUNTLET"
        title_size = cv2.getTextSize(title, HUD_FONT, 0.6, 2)[0]
        title_x = (w - title_size[0]) // 2
        cv2.putText(output, title, (title_x, 25),
                    HUD_FONT, 0.6, HUD_COLOR_ACCENT, 2, cv2.LINE_AA)

        # ── Debug info ──
        if self.debug_mode and self.state in (STATE_READY_TO_SNAP, STATE_COLLECTING):
            debug = self.snap_det.get_debug_info()
            debug_y = 50
            debug_items = [
                f"Snap State: {debug['state']}",
                f"Pinch Dist: {debug['distance']:.1f}px",
                f"Velocity:   {debug['velocity']:.1f}px/f",
                f"Armed:      {debug['armed']}",
                f"App State:  {self.state}",
            ]
            for item in debug_items:
                cv2.putText(output, item, (12, debug_y),
                            HUD_FONT_PLAIN if hasattr(cv2, 'FONT_HERSHEY_PLAIN') else HUD_FONT_SMALL,
                            1.0, (0, 200, 200), 1, cv2.LINE_AA)
                debug_y += 18

        return output

    def _update_fps(self):
        """Update the FPS counter."""
        self._fps_counter += 1
        now = time.time()
        elapsed = now - self._fps_timer
        if elapsed >= 1.0:
            self.fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_timer = now

    def _cleanup(self):
        """Release all resources."""
        print("\n  Cleaning up...")
        self.cap.release()
        self.tracker.release()
        self.disintegration.release()
        cv2.destroyAllWindows()
        if AUDIO_AVAILABLE:
            try:
                pygame.mixer.quit()
            except Exception:
                pass
        print("  Done. Goodbye!\n")




def main():
    """Entry point."""
    try:
        app = InfinityGauntletApp()
        app.run()
    except RuntimeError as e:
        print(f"\n  Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
