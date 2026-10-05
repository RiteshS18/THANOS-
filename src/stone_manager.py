"""
stone_manager.py — Manages the 6 Infinity Stones: placement, collection, and animations.

Renders floating stones at fixed screen positions with idle bobbing animation.
Detects when the user's hand is close enough to collect each stone, triggers
a fly-to-gauntlet animation, and tracks collection state.
"""

import math
import time
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

# ─────────────────────────── Configurable Constants ───────────────────────────
COLLECTION_DISTANCE = 60        # Pixels — how close hand must be to collect a stone
STONE_RADIUS = 20               # Base radius of floating stone
STONE_GLOW_RADIUS = 38          # Outer glow radius
FLOAT_AMPLITUDE = 10            # Pixels of vertical bobbing
FLOAT_SPEED = 0.06              # Radians per frame for bobbing
PULSE_SPEED = 0.08              # Radians per frame for size pulsing
PULSE_AMPLITUDE = 3             # Pixels of radius pulsing
COLLECTION_ANIM_FRAMES = 18     # Frames for fly-to-gauntlet animation
STONE_BORDER_THICKNESS = 2      # Border around each floating stone
# ──────────────────────────────────────────────────────────────────────────────

# Stone definitions: name → (BGR color, display label)
STONE_DEFS = {
    "power":   {"color": (200, 50, 180),   "label": "Power",   "order": 0},
    "space":   {"color": (230, 150, 30),   "label": "Space",   "order": 1},
    "reality": {"color": (50, 50, 230),    "label": "Reality", "order": 2},
    "soul":    {"color": (60, 150, 255),   "label": "Soul",    "order": 3},
    "mind":    {"color": (30, 230, 230),   "label": "Mind",    "order": 4},
    "time":    {"color": (60, 200, 60),    "label": "Time",    "order": 5},
}

# Landmark indices for stone slots on the gauntlet (used for animation target)
STONE_SLOT_LANDMARKS = {
    "power":   5,
    "space":   9,
    "reality": 13,
    "soul":    17,
    "mind":    0,
    "time":    2,
}


class StoneManager:
    """
    Manages floating Infinity Stones and their collection by the user's hand.

    Stones float at fixed positions along the top-right arc of the frame.
    When the user's hand moves close enough, the stone animates flying toward
    its gauntlet slot and is marked as collected.
    """

    def __init__(self, frame_width: int, frame_height: int):
        self.frame_w = frame_width
        self.frame_h = frame_height
        self._frame_count = 0

        # Initialize stone state
        self.stones: Dict[str, dict] = {}
        self._init_stone_positions()

        # Collection animation state
        self._anim_active: Dict[str, dict] = {}  # stone_name → anim data

    def _init_stone_positions(self):
        """Place 6 stones in an arc along the top-right area of the frame."""
        # Arc center and radius
        arc_cx = self.frame_w - 100
        arc_cy = 60
        arc_radius = 140
        start_angle = math.pi * 0.7   # Start from upper-left of arc
        end_angle = math.pi * 1.8     # Sweep to lower-right

        sorted_stones = sorted(STONE_DEFS.items(), key=lambda x: x[1]["order"])

        for i, (name, defn) in enumerate(sorted_stones):
            t = i / max(1, len(sorted_stones) - 1)
            angle = start_angle + t * (end_angle - start_angle)
            base_x = int(arc_cx + arc_radius * math.cos(angle))
            base_y = int(arc_cy + arc_radius * math.sin(angle))

            # Clamp to frame bounds with margin
            base_x = max(STONE_GLOW_RADIUS + 10, min(self.frame_w - STONE_GLOW_RADIUS - 10, base_x))
            base_y = max(STONE_GLOW_RADIUS + 10, min(self.frame_h - STONE_GLOW_RADIUS - 10, base_y))

            self.stones[name] = {
                "color": defn["color"],
                "label": defn["label"],
                "base_pos": (base_x, base_y),
                "collected": False,
                "phase_offset": i * 1.2,  # Stagger bobbing phase
            }

    def update(
        self,
        palm_center: Optional[Tuple[int, int]],
        finger_tips: Optional[Dict[str, Tuple[int, int]]],
        landmarks: Optional[List[Tuple[int, int]]] = None,
    ) -> Optional[str]:
        """
        Check for stone collection and update animations.

        Args:
            palm_center: (x, y) of palm center, or None if no hand.
            finger_tips: Dict of fingertip positions, or None.
            landmarks: Full 21-landmark list for animation targets.

        Returns:
            Name of stone just collected this frame, or None.
        """
        self._frame_count += 1
        collected_this_frame = None

        if palm_center is not None:
            # Check proximity to each uncollected stone
            for name, stone in self.stones.items():
                if stone["collected"]:
                    continue
                if name in self._anim_active:
                    continue  # Already animating

                sx, sy = self._get_float_pos(name)
                dist = math.hypot(palm_center[0] - sx, palm_center[1] - sy)

                # Also check fingertip distances (any fingertip touching counts)
                min_dist = dist
                if finger_tips:
                    for tip_pos in finger_tips.values():
                        d = math.hypot(tip_pos[0] - sx, tip_pos[1] - sy)
                        min_dist = min(min_dist, d)

                if min_dist < COLLECTION_DISTANCE:
                    # Start collection animation
                    target = (0, 0)
                    if landmarks and name in STONE_SLOT_LANDMARKS:
                        slot_idx = STONE_SLOT_LANDMARKS[name]
                        target = landmarks[slot_idx]
                    else:
                        target = palm_center

                    self._anim_active[name] = {
                        "start_pos": (sx, sy),
                        "target_pos": target,
                        "frame": 0,
                        "total_frames": COLLECTION_ANIM_FRAMES,
                    }
                    collected_this_frame = name

        # Update active animations
        finished = []
        for name, anim in self._anim_active.items():
            anim["frame"] += 1
            if anim["frame"] >= anim["total_frames"]:
                self.stones[name]["collected"] = True
                finished.append(name)
        for name in finished:
            del self._anim_active[name]

        return collected_this_frame

    def render(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw all uncollected floating stones and any active collection animations.
        """
        # Draw uncollected stones
        for name, stone in self.stones.items():
            if stone["collected"]:
                continue
            if name in self._anim_active:
                # Draw animation instead
                self._draw_collection_anim(frame, name)
                continue

            # Floating position with bobbing
            fx, fy = self._get_float_pos(name)
            color = stone["color"]

            # Pulsing radius
            pulse = PULSE_AMPLITUDE * math.sin(
                self._frame_count * PULSE_SPEED + stone["phase_offset"]
            )
            radius = int(STONE_RADIUS + pulse)

            # Draw outer glow
            self._draw_glow(frame, (fx, fy), color, STONE_GLOW_RADIUS)

            # Draw stone body
            cv2.circle(frame, (fx, fy), radius, color, -1, cv2.LINE_AA)

            # Dark border
            cv2.circle(frame, (fx, fy), radius, (30, 30, 30),
                       STONE_BORDER_THICKNESS, cv2.LINE_AA)

            # Specular highlight
            hx, hy = fx - radius // 3, fy - radius // 3
            cv2.circle(frame, (hx, hy), max(2, radius // 4),
                       (255, 255, 255), -1, cv2.LINE_AA)

            # Label below stone
            label = stone["label"]
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            tx = fx - text_size[0] // 2
            ty = fy + radius + 18
            cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (220, 220, 220), 1, cv2.LINE_AA)

        return frame

    def get_collection_state(self) -> Dict[str, bool]:
        """Return dict mapping stone name → collected bool."""
        return {name: s["collected"] for name, s in self.stones.items()}

    def all_collected(self) -> bool:
        """Check if all 6 stones are collected."""
        return all(s["collected"] for s in self.stones.values())

    def collected_count(self) -> int:
        """Count of collected stones."""
        return sum(1 for s in self.stones.values() if s["collected"])

    def reset(self):
        """Reset all stones to uncollected state."""
        for stone in self.stones.values():
            stone["collected"] = False
        self._anim_active.clear()
        self._frame_count = 0

    # ──────────────────────── Private Helpers ────────────────────────

    def _get_float_pos(self, name: str) -> Tuple[int, int]:
        """Get the current floating position of a stone (with bobbing)."""
        stone = self.stones[name]
        bx, by = stone["base_pos"]
        bob = FLOAT_AMPLITUDE * math.sin(
            self._frame_count * FLOAT_SPEED + stone["phase_offset"]
        )
        return (bx, int(by + bob))

    def _draw_glow(
        self, frame: np.ndarray, center: Tuple[int, int], color: Tuple[int, int, int],
        radius: int
    ):
        """Draw a soft radial glow effect."""
        glow_overlay = frame.copy()
        for r in range(radius, STONE_RADIUS, -2):
            alpha = 0.06 * (1.0 - (r - STONE_RADIUS) / (radius - STONE_RADIUS))
            cv2.circle(glow_overlay, center, r, color, 2, cv2.LINE_AA)
        cv2.addWeighted(glow_overlay, 0.3, frame, 0.7, 0, frame)

    def _draw_collection_anim(self, frame: np.ndarray, name: str):
        """Draw a stone flying from its position toward the gauntlet slot."""
        anim = self._anim_active[name]
        t = anim["frame"] / anim["total_frames"]
        # Ease-in-out cubic
        t_ease = t * t * (3 - 2 * t)

        sx, sy = anim["start_pos"]
        tx, ty = anim["target_pos"]
        cx = int(sx + (tx - sx) * t_ease)
        cy = int(sy + (ty - sy) * t_ease)

        # Shrinking radius and increasing glow during animation
        radius = int(STONE_RADIUS * (1 - t_ease * 0.6))
        color = self.stones[name]["color"]

        # Bright flash during animation
        flash_color = tuple(min(255, c + int(100 * (1 - t))) for c in color)

        cv2.circle(frame, (cx, cy), radius + 4, flash_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), radius, color, -1, cv2.LINE_AA)

        # Trail effect: draw fading circles along the path
        for trail_t in np.linspace(max(0, t - 0.15), t, 4):
            trail_ease = trail_t * trail_t * (3 - 2 * trail_t)
            trail_x = int(sx + (tx - sx) * trail_ease)
            trail_y = int(sy + (ty - sy) * trail_ease)
            trail_r = max(1, int(radius * 0.4 * (trail_t / max(t, 0.01))))
            cv2.circle(frame, (trail_x, trail_y), trail_r, flash_color, -1, cv2.LINE_AA)
