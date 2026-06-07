# Best-Run Video Detector

Local web UI plus FastAPI backend for video deepfake detection.
The inference runtime and required local model assets are vendored under `backend/reference_model`.

For fresh setup, see [HOW_TO.md](HOW_TO.md).

## Run Backend

```bash
cd /home/comp/face_detect_app
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m backend.main
```

The inference config requests CUDA. Backend now fails fast if CUDA is not visible.
Use CPU only for debugging:

```bash
FACE_DETECT_ALLOW_CPU=1 ./.venv/bin/python -m backend.main
```

Health:

```bash
curl http://127.0.0.1:8765/health
```

## Open Web UI

Open the local web UI directly:

```text
/home/comp/face_detect_app/index.html
```

Or serve it over localhost:

```bash
cd /home/comp/face_detect_app
./.venv/bin/python -m http.server 8080
```

Then open:

```text
http://127.0.0.1:8080/
```

Choose a local video file or refresh backend videos, then play the video to run analysis.

Supported models:

- `canonical_best`: default checkpoint.
- `ffpp_celebdf`: combined-domain checkpoint trained with FF++ C23 and CelebDF data.

Local copied weights live under:

```text
backend/model_weights/canonical_best/best.pt
backend/model_weights/ffpp_celebdf/best.pt
```

## API

- `GET /health`: returns model metadata and load status.
- `POST /v1/predict_frames`: accepts exactly 32 JPEG/PNG multipart files under `frames`
  and optional multipart field `model_id`.

Response includes `Real` / `Fake`, fake probability, confidence, threshold, and top generator prediction.

No frames or videos are written to disk by default.
