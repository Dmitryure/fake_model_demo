# Setup

## Backend

```bash
cd /home/comp/face_detect_app
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m backend.main
```

Backend listens on `http://127.0.0.1:8765`.

## Web UI

Open the web UI directly:

```text
/home/comp/face_detect_app/index.html
```

Or serve it over localhost:

```bash
cd /home/comp/face_detect_app
./.venv/bin/python -m http.server 8080
```

Then open `http://127.0.0.1:8080/`.

Choose a local video file or refresh backend videos, then play the video to run analysis.
