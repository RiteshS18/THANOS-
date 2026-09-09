# Snap to Dust

A local real-time computer-vision effect: a webcam person snaps their fingers, freezes briefly, dissolves into colored dust, disappears, and optionally reconstructs.

## Stages

1. **Webcam:** OpenCV captures and mirrors a configurable camera stream.
2. **YOLO person segmentation:** Ultralytics selects the largest class-0 person mask and cleans it with morphology.
3. **MediaPipe hand tracking:** Hand landmarks are processed in real time.
4. **Finger snap detection:** A temporal state machine measures normalized thumb/index distance, approach speed, and rapid separation.
5. **Particle generation:** Person pixels are sampled into a bounded NumPy particle buffer.
6. **Particle physics:** Wind, gravity, turbulence, drag, randomized lifetime, and opacity create the dust motion.
7. **Disintegration:** The captured person area is inpainted into a clean background, then particles render above it.
8. **Reappearance:** Particles converge toward their original pixels and the live camera returns.
9. **Performance:** YOLO is run every few frames, the model uses a 640px inference size, and particle state is vectorized.
10. **Polished loop:** HUD, flash, shake, dust glow, reset, error messages, and graceful cleanup are included.

## Install

Python 3.10 or 3.11 is recommended. From this directory:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` in that terminal, or activate with `venv\\Scripts\\activate.bat` from Command Prompt.

Download the lightweight segmentation model into `models/`:

```powershell
python -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt')"
Move-Item yolov8n-seg.pt models/yolov8n-seg.pt
```

The hand tracker uses the current MediaPipe Tasks API. On first startup it automatically downloads `models/hand_landmarker.task` when that file is absent.

The application also asks Ultralytics for `yolov8n-seg.pt` automatically when the configured file is absent, provided the machine has internet access. GPU acceleration is used automatically when the installed PyTorch build supports CUDA.

## Run

```powershell
python main.py
```

Stand about 1.5 to 3 meters from the camera with your full body visible. Hold one hand near your upper chest or face, bring the thumb and index finger together, then separate them quickly as in a normal finger snap. The detector is intentionally exposed in the HUD so you can tune thresholds in `config.py` for your camera and lighting.

Press `r` to reset the effect. Press `q` or `Esc` to quit.

## Debugging and configuration

Set `DEBUG_MODE = True` in `config.py` to show snap state, normalized thumb-index distance, velocity, FPS, inference time, particle count, and current effect state. Set `SHOW_MASK = True` to inspect the YOLO mask. The default `ENABLE_REAPPEAR = False` holds on the clean background after the dust fades; set it to `True` and adjust `REAPPEAR_DELAY` to enable reconstruction. Important controls include `CAMERA_INDEX`, `FRAME_WIDTH`, `FRAME_HEIGHT`, `MAX_PARTICLES`, `DUST_DURATION`, `ENABLE_REAPPEAR`, and all `SNAP_*` thresholds.

For a slower CPU, reduce `FRAME_WIDTH`, `FRAME_HEIGHT`, `YOLO_IMGSZ`, `MAX_PARTICLES`, or increase `SEGMENTATION_INTERVAL`. For several people, the largest detected person is selected.

## Common errors

- **Webcam unavailable:** Close Zoom, Teams, OBS, or other camera users; check Windows camera permissions; try `CAMERA_INDEX = 1`.
- **Model missing/download fails:** Download `yolov8n-seg.pt` manually and place it at `models/yolov8n-seg.pt`.
- **MediaPipe unavailable:** Re-run `pip install -r requirements.txt` inside the active virtual environment. This project uses MediaPipe Tasks (`mediapipe>=1.0.1`), not the removed `mp.solutions.hands` API. Ensure the first run has internet access so `hand_landmarker.task` can download.
- **No person mask:** Improve lighting, step farther back, and keep the full body inside the frame.
- **Snap does not trigger:** Use the debug distance and velocity values, then adjust `SNAP_CLOSE_DISTANCE`, `SNAP_RELEASE_DISTANCE`, and `SNAP_SEPARATION_SPEED`.
- **Low FPS:** Lower resolution and particle count first; then increase `SEGMENTATION_INTERVAL`.

The `assets/` directory is reserved for an optional `snap.wav`; sound is disabled by default and the current implementation remains fully usable without it.
