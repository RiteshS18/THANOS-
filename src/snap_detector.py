"""
snap_detector.py — Gesture recognition for the Thanos snap.

Detects a snap gesture defined as the thumb tip and middle fingertip
rapidly coming together (pinch) from an open-hand state. Requires the
gesture to hold for a few consecutive frames to avoid false positives.
Only triggers when armed (all stones collected).
"""

import math
from collections import deque
from typing import Dict, Tuple, Optional

# ─────────────────────────── Configurable Constants ───────────────────────────
PINCH_THRESHOLD = 35            # Pixels — distance below which thumb+middle = pinched
OPEN_THRESHOLD = 80             # Pixels — distance above which hand is considered open
SNAP_WINDOW = 12                # Frames — max time to go from open to pinch
CONFIRM_FRAMES = 3              # Consecutive pinch frames needed to confirm snap
COOLDOWN_FRAMES = 90            # Frames to wait after snap before re-arming
PINCH_VELOCITY_THRESHOLD = 8    # Pixels/frame — minimum closing speed to count
# ──────────────────────────────────────────────────────────────────────────────


class SnapDetector:
    """
    Detects the Thanos snap gesture (thumb + middle finger pinch).

    The detector tracks the distance between thumb tip (landmark 4) and
    middle fingertip (landmark 12) over time. A snap is detected when:
    1. The hand transitions from open (distance > OPEN_THRESHOLD) to
       pinched (distance < PINCH_THRESHOLD) within SNAP_WINDOW frames.
    2. The closing velocity exceeds PINCH_VELOCITY_THRESHOLD.
    3. The pinch state holds for CONFIRM_FRAMES consecutive frames.

    After triggering, the detector enters a cooldown period.
    """

    def __init__(self):
        self._armed = False          # Whether detector is accepting gestures
        self._cooldown_counter = 0   # Frames remaining in cooldown
        self._confirm_counter = 0    # Consecutive pinch frames
        self._was_open = False       # Whether hand was recently open
        self._open_frame = 0         # Frame when hand was last seen open
        self._frame_count = 0        # Total frames processed

        # Distance history for velocity computation
        self._dist_history = deque(maxlen=10)

        # Debug state
        self._current_dist = 0.0
        self._current_velocity = 0.0
        self._state_label = "IDLE"

    def update(self, finger_tips: Optional[Dict[str, Tuple[int, int]]]) -> bool:
        """
        Process one frame of finger positions.

        Args:
            finger_tips: Dict with at least "thumb" and "middle" keys mapping
                         to (x, y) pixel positions. None if no hand detected.

        Returns:
            True if a snap was just confirmed this frame, False otherwise.
        """
        self._frame_count += 1

        # Handle cooldown
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            self._state_label = f"COOLDOWN ({self._cooldown_counter})"
            return False

        # No hand or not armed → nothing to do
        if finger_tips is None or not self._armed:
            self._state_label = "WAITING" if not self._armed else "NO HAND"
            self._confirm_counter = 0
            self._was_open = False
            return False

        # Get thumb and middle fingertip positions
        thumb = finger_tips.get("thumb")
        middle = finger_tips.get("middle")
        if thumb is None or middle is None:
            self._state_label = "MISSING TIPS"
            return False

        # Compute distance
        dist = math.hypot(thumb[0] - middle[0], thumb[1] - middle[1])
        self._current_dist = dist
        self._dist_history.append(dist)

        # Compute velocity (pixels/frame, negative = closing)
        velocity = 0.0
        if len(self._dist_history) >= 2:
            velocity = self._dist_history[-2] - self._dist_history[-1]  # positive = closing
        self._current_velocity = velocity

        # State machine: track open → pinch transition
        if dist > OPEN_THRESHOLD:
            self._was_open = True
            self._open_frame = self._frame_count
            self._confirm_counter = 0
            self._state_label = "OPEN"
            return False

        # Check if we're in a valid snap window
        frames_since_open = self._frame_count - self._open_frame

        if dist < PINCH_THRESHOLD and self._was_open:
            if frames_since_open <= SNAP_WINDOW:
                # Check velocity
                if velocity >= PINCH_VELOCITY_THRESHOLD or self._confirm_counter > 0:
                    self._confirm_counter += 1
                    self._state_label = f"CONFIRMING ({self._confirm_counter}/{CONFIRM_FRAMES})"

                    if self._confirm_counter >= CONFIRM_FRAMES:
                        # SNAP CONFIRMED!
                        self._trigger_snap()
                        return True
                else:
                    self._state_label = "SLOW PINCH"
            else:
                # Window expired
                self._was_open = False
                self._confirm_counter = 0
                self._state_label = "PINCHED (expired)"
        elif dist < OPEN_THRESHOLD:
            self._state_label = "PARTIAL"
            if self._confirm_counter > 0 and dist >= PINCH_THRESHOLD:
                # Lost the pinch before confirmation
                self._confirm_counter = 0

        return False

    def arm(self):
        """Arm the detector (call when all stones are collected)."""
        self._armed = True

    def disarm(self):
        """Disarm the detector."""
        self._armed = False
        self._confirm_counter = 0
        self._was_open = False

    def is_armed(self) -> bool:
        """Check if the detector is currently armed."""
        return self._armed and self._cooldown_counter == 0

    def reset(self):
        """Full reset of detector state."""
        self._armed = False
        self._cooldown_counter = 0
        self._confirm_counter = 0
        self._was_open = False
        self._dist_history.clear()
        self._state_label = "IDLE"

    def get_debug_info(self) -> dict:
        """Return debug information for HUD display."""
        return {
            "armed": self._armed,
            "distance": self._current_dist,
            "velocity": self._current_velocity,
            "state": self._state_label,
            "confirm_progress": self._confirm_counter / CONFIRM_FRAMES if CONFIRM_FRAMES > 0 else 0,
            "cooldown": self._cooldown_counter,
        }

    # ──────────────────────── Private Helpers ────────────────────────

    def _trigger_snap(self):
        """Handle snap trigger: enter cooldown and reset counters."""
        self._cooldown_counter = COOLDOWN_FRAMES
        self._confirm_counter = 0
        self._was_open = False
        self._dist_history.clear()
        self._state_label = "SNAPPED!"
