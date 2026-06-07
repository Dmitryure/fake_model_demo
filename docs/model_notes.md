# Model Notes

Inference runtime is vendored under `backend/reference_model`.

- Best-run checkpoint copies live under `backend/model_weights`.
- Runtime feature extractors, modality branches, and local assets live under `backend/reference_model`.

Backend resolves `backend/inference_config.yaml` asset paths against `backend/reference_model`,
then loads runtime model code and selected checkpoint at startup.

Input policy:

- `POST /v1/predict_frames` requires exactly 32 multipart image fields named `frames`.
- Images decode to RGB arrays.
- Face crop uses OpenCV Haar, detection frequency 16, large box coefficient 1.5.
- If no face is detected, full frame is used.
- Modality frame counts: `rgb=16`, `eye_gaze=32`, `face_mesh=32`, `depth=32`.
- Binary threshold is `0.5`.
- Backend does not persist frames or videos.
