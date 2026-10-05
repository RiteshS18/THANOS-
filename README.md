# 🫰 Infinity Gauntlet: The Snap

A real-time computer vision project that lets you **become Thanos** using just your webcam. Track your hand, collect all 6 Infinity Stones, and snap your fingers to trigger a cinematic dust disintegration effect — all powered by Python, OpenCV, and MediaPipe.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green?style=flat-square)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange?style=flat-square)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🖐️ **Hand Tracking** | Real-time 21-landmark hand detection via MediaPipe Hands |
| 🧤 **Gauntlet Overlay** | Golden Infinity Gauntlet rendered directly onto your hand |
| 💎 **Stone Collection** | 6 floating Infinity Stones to collect by moving your hand near them |
| 🫰 **Snap Detection** | Pinch your thumb + middle finger to trigger the snap |
| 💨 **Dust Disintegration** | Cinematic particle effect — your silhouette breaks apart into drifting ash |
| 🎵 **Sound Effects** | Optional audio for collection chimes and snap boom |
| 📊 **HUD** | On-screen stone counter, status indicator, FPS display |

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+**
- A **webcam** (built-in or external)
- **Windows / macOS / Linux**

### Installation

```bash
# Clone or navigate to the project
cd thanos

# Install dependencies
pip install -r requirements.txt

# Run!
python -m src.main
```

### Alternative run command
```bash
python src/main.py
```

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| **Hand movement** | Move your hand near floating stones to collect them |
| **Thumb + Middle pinch** | Snap gesture (only works after all 6 stones collected) |
| `D` | Toggle debug overlay (shows landmarks, distances, snap state) |
| `R` | Reset — start collecting stones again |
| `Q` | Quit the application |

---

## 🏗️ Architecture

```
Webcam → MediaPipe Hands → Hand Tracker → ┬→ Gauntlet Overlay ──→ Frame
                                           ├→ Stone Manager    ──→ Compositor
                                           ├→ Snap Detector    ──→ ↓
                                           └→ Disintegration   ──→ Display
```

### State Machine

```
COLLECTING ──(all 6 stones)──→ READY_TO_SNAP ──(snap gesture)──→ DISINTEGRATING ──(done)──→ RESET ──→ COLLECTING
```

---

## 📁 Project Structure

```
thanos/
├── assets/
│   └── sfx/                    # Drop .wav files here for audio
│       ├── collect.wav          # (optional) stone collection sound
│       └── snap.wav             # (optional) snap sound effect
├── src/
│   ├── __init__.py
│   ├── hand_tracker.py          # MediaPipe Hands wrapper + landmark utilities
│   ├── gauntlet_overlay.py      # Golden gauntlet rendering on hand
│   ├── stone_manager.py         # 6 floating stones + collection logic
│   ├── snap_detector.py         # Pinch gesture recognition
│   ├── disintegration.py        # Person mask → particle dust effect
│   └── main.py                  # Main loop + state machine + HUD
├── requirements.txt
└── README.md
```

---

## 🔧 How Each Effect Works

### 1. Hand Tracking (`hand_tracker.py`)
- Uses **MediaPipe Hands** to detect 21 landmark points on the hand in real-time
- Computes **palm center** (average of wrist + 4 MCP joints), **fingertip positions**, **hand bounding box**, and **rotation angle**
- Applies **Exponential Moving Average (EMA) smoothing** to reduce landmark jitter
- Key constant: `SMOOTHING_FACTOR = 0.35` (increase for smoother but laggier tracking)

### 2. Gauntlet Overlay (`gauntlet_overlay.py`)
- Draws a **golden polygon** over the palm region using landmark positions as vertices
- Each finger is rendered as a **thick armored line segment** with metallic highlight
- Collected stones appear as **glowing gems** at specific knuckle positions:
  - Power (purple) → Index MCP
  - Space (blue) → Middle MCP
  - Reality (red) → Ring MCP
  - Soul (orange) → Pinky MCP
  - Mind (yellow) → Wrist
  - Time (green) → Thumb CMC
- Radial glow effects use **additive blending** for a supernatural look

### 3. Stone Collection (`stone_manager.py`)
- 6 stones placed in an **arc** across the top-right area of the frame
- Each stone has an **idle bobbing animation** (sine wave) and **pulsing glow**
- When any fingertip or palm center comes within `COLLECTION_DISTANCE` (60px), the stone is collected
- Collection triggers a **fly-to-gauntlet animation** with cubic ease-in-out and trail particles

### 4. Snap Detection (`snap_detector.py`)
- Tracks distance between **thumb tip** (landmark 4) and **middle fingertip** (landmark 12)
- Snap = hand goes from **open** (distance > 80px) to **pinched** (distance < 35px) within 12 frames
- Requires closing **velocity > 8px/frame** to reject slow/accidental pinches
- Must hold pinch for **3 consecutive frames** to confirm (prevents false positives)
- **90-frame cooldown** after each snap to prevent re-triggering

### 5. Disintegration Effect (`disintegration.py`)
- On snap trigger, runs **MediaPipe Selfie Segmentation** to extract person mask
- Divides the person region into a grid of **5×5 pixel blocks**, each becoming a particle
- Each particle gets: position, color (sampled from frame), randomized velocity (upward drift), alpha, size
- **Wave delay** creates the iconic left-to-right dissolve (particles on the left start moving first)
- Physics updated with **vectorized NumPy** operations for performance
- Particles drift upward with turbulence, fade out, and shrink
- Background shows through using a **running average background model**
- Initial **white flash** and "THE SNAP" text overlay for cinematic impact

---

## ⚙️ Tunable Constants

All thresholds are configurable at the top of each source file:

| File | Constant | Default | Description |
|------|----------|---------|-------------|
| `hand_tracker.py` | `SMOOTHING_FACTOR` | 0.35 | Landmark jitter smoothing (0–1) |
| `stone_manager.py` | `COLLECTION_DISTANCE` | 60 | Pixels to collect a stone |
| `stone_manager.py` | `FLOAT_AMPLITUDE` | 10 | Stone bobbing amplitude |
| `snap_detector.py` | `PINCH_THRESHOLD` | 35 | Pinch distance in pixels |
| `snap_detector.py` | `OPEN_THRESHOLD` | 80 | Open hand distance |
| `snap_detector.py` | `CONFIRM_FRAMES` | 3 | Frames to confirm snap |
| `disintegration.py` | `PARTICLE_BLOCK_SIZE` | 5 | Particle resolution |
| `disintegration.py` | `EFFECT_DURATION_FRAMES` | 150 | Disintegration length |
| `disintegration.py` | `WAVE_DELAY_MAX` | 40 | Left-to-right wave spread |
| `main.py` | `FRAME_WIDTH` | 640 | Capture resolution width |
| `main.py` | `FRAME_HEIGHT` | 480 | Capture resolution height |

---

## 🔊 Sound Effects (Optional)

Drop `.wav` files into `assets/sfx/`:
- `collect.wav` — plays when a stone is collected
- `snap.wav` — plays when the snap triggers

If the files are missing or `pygame` isn't installed, the app runs silently.

---

## 🎯 Stretch Goals

- [ ] Stones that follow slow orbiting paths (chase/grab mechanic)
- [ ] Multi-hand support with primary-hand selection
- [ ] Auto-record the snap moment as GIF/MP4
- [ ] Browser version with MediaPipe.js

---

## 📋 Tech Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| Hand Tracking | MediaPipe Hands | 21-landmark real-time detection |
| Person Segmentation | MediaPipe Selfie Segmentation | Person mask for disintegration |
| Video I/O | OpenCV (cv2) | Capture, compositing, display |
| Particle Math | NumPy | Vectorized physics updates |
| Audio | pygame.mixer | Optional sound effects |

---

*Perfectly balanced, as all things should be.* ⚖️
