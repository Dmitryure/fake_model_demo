from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference import ReferenceModelService  # noqa: E402
from backend.settings import DEFAULT_MODEL_ID, EXPECTED_FRAME_COUNT, MODEL_VARIANTS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fake-video inference for one local video file.",
    )
    parser.add_argument("video_path", type=Path, help="Path to a local video file.")
    parser.add_argument(
        "--model-id",
        choices=sorted(MODEL_VARIANTS),
        default=DEFAULT_MODEL_ID,
        help="Checkpoint variant to use.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of text.",
    )
    return parser.parse_args()


def evenly_spaced_indices(total_frames: int, count: int) -> list[int]:
    if total_frames < count:
        raise ValueError(f"Need at least {count} video frames, got {total_frames}.")
    if count <= 1:
        return [0]
    last_index = total_frames - 1
    return [round(index * last_index / (count - 1)) for index in range(count)]


def read_frame_at(capture: Any, index: int) -> Any:
    import cv2

    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame_bgr = capture.read()
    if not ok or frame_bgr is None:
        raise ValueError(f"Could not read video frame at index {index}.")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def capture_evenly_spaced_frames(video_path: Path, count: int = EXPECTED_FRAME_COUNT) -> list[Any]:
    import cv2

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            return capture_first_frames(capture, count)
        return [
            read_frame_at(capture, index)
            for index in evenly_spaced_indices(total_frames, count)
        ]
    finally:
        capture.release()


def capture_first_frames(capture: Any, count: int) -> list[Any]:
    import cv2

    frames = []
    while len(frames) < count:
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    if len(frames) != count:
        raise ValueError(f"Expected at least {count} readable frames, got {len(frames)}.")
    return frames


def result_payload(
    result: Any,
    model_id: str,
    video_path: Path,
    capture_ms: float,
    load_ms: float,
    predict_wall_ms: float,
    wall_total_ms: float,
) -> dict[str, Any]:
    return {
        "video": str(video_path),
        "model_id": model_id,
        "label": result.label,
        "fake_probability": result.fake_probability,
        "real_probability": result.real_probability,
        "confidence": result.confidence,
        "confidence_label": result.confidence_label,
        "top_generator": result.generator_debug.get("top_generator"),
        "timings_ms": {
            "capture": capture_ms,
            "load": load_ms,
            "predict_wall": predict_wall_ms,
            "preprocess": result.timings_ms.get("preprocess"),
            "model_inference": result.timings_ms.get("inference"),
            "predict_total": result.timings_ms.get("total"),
            "wall_total": wall_total_ms,
        },
    }


def print_text(payload: dict[str, Any]) -> None:
    timings = payload["timings_ms"]
    top = payload.get("top_generator")
    top_text = "-"
    if isinstance(top, dict):
        top_text = f"{top.get('name', '-')} ({float(top.get('probability', 0.0)):.4f})"

    print(f"Video: {payload['video']}")
    print(f"Model: {payload['model_id']}")
    print(f"Label: {payload['label']}")
    print(f"Fake probability: {payload['fake_probability']:.4f}")
    print(f"Real probability: {payload['real_probability']:.4f}")
    print(f"Confidence: {payload['confidence']:.4f} ({payload['confidence_label']})")
    print(f"Likely generator: {top_text}")
    print("Timings ms:")
    print(f"  capture: {timings['capture']:.1f}")
    print(f"  load: {timings['load']:.1f}")
    print(f"  preprocess: {timings['preprocess']:.1f}")
    print(f"  model_inference: {timings['model_inference']:.1f}")
    print(f"  predict_total: {timings['predict_total']:.1f}")
    print(f"  predict_wall: {timings['predict_wall']:.1f}")
    print(f"  wall_total: {timings['wall_total']:.1f}")


def main() -> int:
    args = parse_args()
    started = time.perf_counter()

    capture_started = time.perf_counter()
    frames = capture_evenly_spaced_frames(args.video_path)
    capture_ms = (time.perf_counter() - capture_started) * 1000.0

    service = ReferenceModelService()
    try:
        load_started = time.perf_counter()
        service.load(args.model_id)
        load_ms = (time.perf_counter() - load_started) * 1000.0

        predict_started = time.perf_counter()
        result = service.predict(frames)
        predict_wall_ms = (time.perf_counter() - predict_started) * 1000.0
    finally:
        service.close()

    payload = result_payload(
        result=result,
        model_id=args.model_id,
        video_path=args.video_path,
        capture_ms=capture_ms,
        load_ms=load_ms,
        predict_wall_ms=predict_wall_ms,
        wall_total_ms=(time.perf_counter() - started) * 1000.0,
    )
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
