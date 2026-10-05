"""
gauntlet_overlay.py — Renders a golden Infinity Gauntlet overlay on the detected hand.

Draws a metallic golden polygon over the palm and finger regions using the
21 MediaPipe landmarks. Embedded stones appear at knuckle positions as
glowing colored gems when collected.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

# ─────────────────────────── Configurable Constants ───────────────────────────
GAUNTLET_COLOR_BASE = (30, 170, 235)    # Golden color in BGR (deep gold)
GAUNTLET_COLOR_HIGHLIGHT = (60, 210, 255)  # Lighter gold for highlights
GAUNTLET_ALPHA = 0.55                   # Transparency of gauntlet overlay
GAUNTLET_BORDER_COLOR = (15, 120, 180)  # Darker gold for edges
GAUNTLET_BORDER_THICKNESS = 2           # Edge line thickness
STONE_EMBED_RADIUS = 7                  # Radius of embedded stone gems
STONE_GLOW_RADIUS = 16                  # Radius of glow around embedded stones
STONE_GLOW_ALPHA = 0.4                  # Glow transparency
# ──────────────────────────────────────────────────────────────────────────────

# Stone colors (BGR format)
STONE_COLORS = {
    "power":   (200, 50, 180),    # Purple
    "space":   (230, 150, 30),    # Blue
    "reality": (50, 50, 230),     # Red
    "soul":    (60, 150, 255),    # Orange
    "mind":    (30, 230, 230),    # Yellow
    "time":    (60, 200, 60),     # Green
}

# Which landmark each stone embeds at on the gauntlet
STONE_SLOT_LANDMARKS = {
    "power":   5,   # Index MCP
    "space":   9,   # Middle MCP
    "reality": 13,  # Ring MCP
    "soul":    17,  # Pinky MCP
    "mind":    0,   # Wrist
    "time":    2,   # Thumb CMC
}


class GauntletOverlay:
    """
    Renders a golden gauntlet overlay on the detected hand using landmark positions.

    The gauntlet is drawn as a series of filled polygons covering the palm and
    each finger segment. Collected Infinity Stones appear as glowing gems at
    their respective knuckle positions.
    """

    def __init__(self):
        pass

    def render(
        self,
        frame: np.ndarray,
        landmarks: List[Tuple[int, int]],
        collected_stones: Dict[str, bool],
        hand_angle: float = 0.0,
    ) -> np.ndarray:
        """
        Composite the gauntlet overlay onto the frame.

        Args:
            frame: BGR frame to draw on.
            landmarks: List of 21 (x, y) pixel-coordinate tuples.
            collected_stones: Dict mapping stone name → bool (True if collected).
            hand_angle: Hand rotation angle in degrees (currently unused, reserved).

        Returns:
            Frame with gauntlet composited on top.
        """
        overlay = frame.copy()

        # Draw the gauntlet shape (palm + fingers)
        self._draw_gauntlet_shape(overlay, landmarks)

        # Draw embedded stones for collected ones
        self._draw_embedded_stones(overlay, landmarks, collected_stones)

        # Blend overlay onto frame
        result = cv2.addWeighted(overlay, GAUNTLET_ALPHA, frame, 1 - GAUNTLET_ALPHA, 0)

        # Draw stone glows on top (additive, not blended with gauntlet alpha)
        self._draw_stone_glows(result, landmarks, collected_stones)

        # Draw gauntlet border lines on top
        self._draw_gauntlet_borders(result, landmarks)

        return result

    def _draw_gauntlet_shape(
        self, overlay: np.ndarray, landmarks: List[Tuple[int, int]]
    ):
        """Draw filled golden polygon regions covering the hand."""
        # Palm region: polygon through MCP joints + wrist
        palm_indices = [0, 1, 5, 9, 13, 17]
        palm_pts = np.array(
            [landmarks[i] for i in palm_indices], dtype=np.int32
        )
        cv2.fillConvexPoly(overlay, palm_pts, GAUNTLET_COLOR_BASE)

        # Inner palm fill — a slightly brighter region in the center
        inner_indices = [5, 9, 13, 17]
        inner_pts = np.array(
            [landmarks[i] for i in inner_indices], dtype=np.int32
        )
        cv2.fillConvexPoly(overlay, inner_pts, GAUNTLET_COLOR_HIGHLIGHT)

        # Draw each finger as a series of thick line segments (like armor plates)
        finger_chains = [
            [1, 2, 3, 4],       # Thumb
            [5, 6, 7, 8],       # Index
            [9, 10, 11, 12],    # Middle
            [13, 14, 15, 16],   # Ring
            [17, 18, 19, 20],   # Pinky
        ]

        for chain in finger_chains:
            for i in range(len(chain) - 1):
                p1 = landmarks[chain[i]]
                p2 = landmarks[chain[i + 1]]
                # Draw thick rounded line to simulate finger armor
                thickness = max(8, self._segment_thickness(p1, p2))
                cv2.line(overlay, p1, p2, GAUNTLET_COLOR_BASE, thickness, cv2.LINE_AA)

                # Add a brighter center line for metallic look
                cv2.line(
                    overlay, p1, p2, GAUNTLET_COLOR_HIGHLIGHT,
                    max(2, thickness // 3), cv2.LINE_AA
                )

            # Draw fingertip cap (circle at the tip)
            tip = landmarks[chain[-1]]
            cv2.circle(overlay, tip, max(5, self._avg_finger_width(landmarks) // 2),
                       GAUNTLET_COLOR_BASE, -1, cv2.LINE_AA)

    def _draw_gauntlet_borders(
        self, frame: np.ndarray, landmarks: List[Tuple[int, int]]
    ):
        """Draw border lines along the gauntlet edges for definition."""
        # Palm border
        palm_indices = [0, 1, 5, 9, 13, 17]
        palm_pts = np.array(
            [landmarks[i] for i in palm_indices], dtype=np.int32
        ).reshape((-1, 1, 2))
        cv2.polylines(frame, [palm_pts], True, GAUNTLET_BORDER_COLOR,
                       GAUNTLET_BORDER_THICKNESS, cv2.LINE_AA)

        # Finger segment borders
        finger_chains = [
            [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
            [13, 14, 15, 16], [17, 18, 19, 20],
        ]
        for chain in finger_chains:
            for i in range(len(chain) - 1):
                p1 = landmarks[chain[i]]
                p2 = landmarks[chain[i + 1]]
                cv2.line(frame, p1, p2, GAUNTLET_BORDER_COLOR,
                         GAUNTLET_BORDER_THICKNESS, cv2.LINE_AA)

    def _draw_embedded_stones(
        self,
        overlay: np.ndarray,
        landmarks: List[Tuple[int, int]],
        collected_stones: Dict[str, bool],
    ):
        """Draw collected stones as filled gems at their slot positions."""
        for stone_name, is_collected in collected_stones.items():
            if not is_collected:
                continue
            if stone_name not in STONE_SLOT_LANDMARKS:
                continue

            slot_idx = STONE_SLOT_LANDMARKS[stone_name]
            pos = landmarks[slot_idx]
            color = STONE_COLORS.get(stone_name, (255, 255, 255))

            # Draw gem: outer dark ring + inner color + white specular highlight
            cv2.circle(overlay, pos, STONE_EMBED_RADIUS + 2, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(overlay, pos, STONE_EMBED_RADIUS, color, -1, cv2.LINE_AA)
            # Specular highlight (small white dot offset up-left)
            highlight = (pos[0] - 2, pos[1] - 2)
            cv2.circle(overlay, highlight, max(2, STONE_EMBED_RADIUS // 3),
                       (255, 255, 255), -1, cv2.LINE_AA)

    def _draw_stone_glows(
        self,
        frame: np.ndarray,
        landmarks: List[Tuple[int, int]],
        collected_stones: Dict[str, bool],
    ):
        """Draw radial glow around each embedded stone (additive blend)."""
        h, w = frame.shape[:2]
        glow_layer = np.zeros_like(frame, dtype=np.float32)

        any_glow = False
        for stone_name, is_collected in collected_stones.items():
            if not is_collected:
                continue
            if stone_name not in STONE_SLOT_LANDMARKS:
                continue

            any_glow = True
            slot_idx = STONE_SLOT_LANDMARKS[stone_name]
            cx, cy = landmarks[slot_idx]
            color = STONE_COLORS.get(stone_name, (255, 255, 255))

            # Draw a soft radial glow using multiple concentric circles with decreasing alpha
            for r in range(STONE_GLOW_RADIUS, 0, -1):
                alpha = STONE_GLOW_ALPHA * (1.0 - r / STONE_GLOW_RADIUS) ** 2
                glow_color = tuple(c * alpha for c in color)
                cv2.circle(glow_layer, (cx, cy), r, glow_color, 2, cv2.LINE_AA)

        if any_glow:
            # Additive blend glow onto frame
            glow_uint8 = np.clip(glow_layer, 0, 255).astype(np.uint8)
            cv2.add(frame, glow_uint8, frame)

    @staticmethod
    def _segment_thickness(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        """Compute line thickness proportional to segment length."""
        dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        return int(max(6, min(20, dist * 0.35)))

    @staticmethod
    def _avg_finger_width(landmarks: List[Tuple[int, int]]) -> int:
        """Estimate finger width from MCP spacing."""
        dists = []
        mcp_indices = [5, 9, 13, 17]
        for i in range(len(mcp_indices) - 1):
            p1 = landmarks[mcp_indices[i]]
            p2 = landmarks[mcp_indices[i + 1]]
            dists.append(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        return int(np.mean(dists)) if dists else 12
