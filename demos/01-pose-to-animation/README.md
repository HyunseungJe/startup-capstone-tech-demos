# Pose to Animation · Python

Goal: Video → model adapter → common pose JSON → skeleton animation.

## Run (uv)

```powershell
cd demos/01-pose-to-animation
uv sync
uv run streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open http://127.0.0.1:8501. uv manages the local `.venv` and `uv.lock`. If uv is not on PATH, use `& "$env:USERPROFILE\.local\bin\uv.exe"` in PowerShell.

Input: uploaded video, shared `../../sample-data/videos/` clips, or a `pose2d/v1` JSON file.
Output: `pose2d/v1`, `pose2d-animation/v1`, H.264 skeleton MP4 (30 fps, longest side 640px, no audio).

Only `src/adapters/mediapipe_pose.py` knows the model. Adapter contract: `initialize()`, `estimate(rgb, timestamp_ms)`, `close()`. Video extraction accepts an adapter instance. Pose and animation modules are model-independent.

13 named joints, normalized image coordinates (+x right, +y down), seconds, null for missing joints, linear interpolation. Bone positions and lengths use source pixels; rotations use clockwise radians. Extraction timestamps follow selected source frames, capped at source FPS.

Status: Prototype. Single person, constant-frame-rate video assumption, no jitter filtering or character attachment. Source and preview players are independent. First inference downloads Google's model into `.cache/`. Local uploads are processed on the Python server.

Minimal real-video check: `uv run python smoke.py` processes walking, squat and sword clips and writes JSON/MP4 to ignored `outputs/`. Detection counts are not accuracy scores.

API: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python
