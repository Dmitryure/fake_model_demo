from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse

from .inference import ReferenceModelService
from .preprocessing import decode_image_bytes, validate_frame_count
from .settings import DEFAULT_MODEL_ID, EXPECTED_FRAME_COUNT, VIDEO_DIR
from .videos import (
    NO_STORE_HEADERS,
    ensure_video_dir,
    iter_video_bytes,
    list_video_files,
    parse_range_header,
    resolve_video_path,
    video_media_type,
    video_stream_headers,
)


ensure_video_dir(VIDEO_DIR)
service = ReferenceModelService()
logger = logging.getLogger("best_run_detector")


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.load()
    try:
        yield
    finally:
        service.close()


app = FastAPI(title="Best-Run Video Detector Backend", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.mount("/videos", StaticFiles(directory=VIDEO_DIR), name="videos")


@app.middleware("http")
async def disable_video_cache(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith(("/videos", "/v1/videos", "/v1/video_stream")):
        response.headers.update(NO_STORE_HEADERS)
    return response


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "model": service.metadata()}


@app.get("/v1/videos")
def videos(response: Response) -> dict[str, object]:
    response.headers.update(NO_STORE_HEADERS)
    return {"videos": list_video_files(VIDEO_DIR)}


@app.get("/v1/video_stream/{filename}")
def stream_video(filename: str, request: Request) -> StreamingResponse:
    try:
        path = resolve_video_path(VIDEO_DIR, filename)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    file_size = path.stat().st_size
    try:
        video_range = parse_range_header(request.headers.get("range"), file_size)
    except ValueError as exc:
        headers = {**NO_STORE_HEADERS, "Content-Range": f"bytes */{file_size}"}
        raise HTTPException(status_code=416, detail=str(exc), headers=headers) from exc

    return StreamingResponse(
        iter_video_bytes(path, video_range.start, video_range.end),
        status_code=video_range.status_code,
        media_type=video_media_type(path),
        headers=video_stream_headers(video_range, file_size),
    )


@app.post("/v1/predict_frames")
async def predict_frames(
    frames: list[UploadFile] = File(...),
    model_id: str = Form(DEFAULT_MODEL_ID),
) -> dict[str, object]:
    started = time.perf_counter()
    logger.info("predict_frames received frame_count=%s model_id=%s", len(frames), model_id)
    try:
        validate_frame_count(len(frames))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    decoded = []
    for index, upload in enumerate(frames):
        if upload.content_type not in {"image/jpeg", "image/png", "application/octet-stream"}:
            raise HTTPException(
                status_code=400,
                detail=f"Frame {index} must be JPEG or PNG, got {upload.content_type}.",
            )
        payload = await upload.read()
        try:
            decoded.append(decode_image_bytes(payload))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Frame {index}: {exc}") from exc

    try:
        service.load(model_id)
        result = service.predict(decoded)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("predict_frames failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    logger.info(
        "predict_frames completed label=%s fake_probability=%.4f total_ms=%.1f",
        result.label,
        result.fake_probability,
        (time.perf_counter() - started) * 1000.0,
    )

    return {
        "label": result.label,
        "fake_probability": result.fake_probability,
        "real_probability": result.real_probability,
        "confidence": result.confidence,
        "confidence_label": result.confidence_label,
        "threshold": result.threshold,
        "debug": {
            "generator": result.generator_debug,
            "preprocessing": result.preprocessing,
            "timings_ms": result.timings_ms,
            "frame_count": EXPECTED_FRAME_COUNT,
            "model_id": service.model_id,
            "model_name": service.paths.display_name,
        },
    }
