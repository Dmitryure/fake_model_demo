# How To Run

Minimal steps for a fresh local checkout.

## What You Need

- Python 3.12 or close equivalent.
- Local virtual environment for this repo.
- CUDA-capable GPU for normal use. CPU mode is only for debugging.
- Vendored runtime code under `backend/reference_model/`.
- Shared model assets under `backend/assets/`.
- Model checkpoints under `backend/model_weights/`.

This app imports model runtime code from `backend/reference_model`, resolves shared assets from `backend/assets`, and loads selected checkpoints from `backend/model_weights`.

## Restore Model Files

Heavy model files are intentionally ignored by git. Download or copy `model_files.zip` into the repo root, then unzip it:

```bash
unzip model_files.zip
```

After unzip, these paths must exist:

```text
backend/assets/checkpoints/mvit_v2_s-ae3be167.pth
backend/assets/models/face_landmarker_v2_with_blendshapes.task
backend/assets/models/depth-anything-v2-small-hf/config.json
backend/assets/models/depth-anything-v2-small-hf/model.safetensors
backend/assets/models/depth-anything-v2-small-hf/preprocessor_config.json
backend/model_weights/canonical_best/best.pt
backend/model_weights/canonical_best/run_config.json
backend/model_weights/ffpp_celebdf/best.pt
backend/model_weights/ffpp_celebdf/run_config.json
```

## Install

```bash
cd path/to/face_detect_app
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## Start Backend

```bash
./.venv/bin/python -m backend.main
```

Backend runs at:

```text
http://127.0.0.1:8765
```

Check it:

```bash
curl http://127.0.0.1:8765/health
```

CPU debug mode:

```bash
FACE_DETECT_ALLOW_CPU=1 ./.venv/bin/python -m backend.main
```

## Open Web UI

Direct file:

```text
index.html
```

Or local server:

```bash
./.venv/bin/python -m http.server 8080
```

Then open:

```text
http://127.0.0.1:8080/
```

## Use App

1. Start backend.
2. Open web UI.
3. Choose a local video file, or put videos in `backend/videos/` and click `Refresh videos`.
4. Click `Analyze video`.
5. UI captures 32 frames, sends them to backend, then shows `Real` or `Fake` plus likely generator.

## Run One Video From CLI

```bash
./.venv/bin/python scripts/predict_video.py path/to/video.mp4
```

Choose the second checkpoint:

```bash
./.venv/bin/python scripts/predict_video.py path/to/video.mp4 --model-id ffpp_celebdf
```

JSON output:

```bash
./.venv/bin/python scripts/predict_video.py path/to/video.mp4 --json
```

## Run A Folder From CLI

```bash
./.venv/bin/python scripts/batch_predict_video.py path/to/video_folder
```

Search subfolders too:

```bash
./.venv/bin/python scripts/batch_predict_video.py path/to/video_folder --recursive
```

Use the second checkpoint:

```bash
./.venv/bin/python scripts/batch_predict_video.py path/to/video_folder --model-id ffpp_celebdf
```

JSON output:

```bash
./.venv/bin/python scripts/batch_predict_video.py path/to/video_folder --json
```

## Test

```bash
./.venv/bin/python -m pytest
```

## Common Problems

- `Config requires CUDA`: GPU/CUDA not visible. Fix CUDA or use CPU debug mode.
- `Backend error`: check terminal running backend.
- No backend videos: create `backend/videos/` and put `.mp4`, `.webm`, `.mov`, `.m4v`, or `.ogg` files there.
- Missing model files: confirm `backend/reference_model` contains runtime code, `backend/assets` contains shared model assets, `backend/inference_config.yaml` exists, and `backend/model_weights/*` contains `best.pt` plus `run_config.json`.
