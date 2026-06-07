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
from backend.settings import DEFAULT_MODEL_ID, MODEL_VARIANTS  # noqa: E402
from scripts.predict_video import (  # noqa: E402
    capture_evenly_spaced_frames,
    result_payload,
)


VIDEO_EXTENSIONS = frozenset({".mp4", ".webm", ".mov", ".m4v", ".ogg", ".avi", ".mkv"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fake-video inference for every supported video in a folder.",
    )
    parser.add_argument("video_dir", type=Path, help="Folder containing local video files.")
    parser.add_argument(
        "--model-id",
        choices=sorted(MODEL_VARIANTS),
        default=DEFAULT_MODEL_ID,
        help="Checkpoint variant to use.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subfolders recursively.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of text rows.",
    )
    return parser.parse_args()


def list_video_files(video_dir: Path, recursive: bool = False) -> list[Path]:
    if not video_dir.exists():
        raise FileNotFoundError(f"Folder not found: {video_dir}")
    if not video_dir.is_dir():
        raise NotADirectoryError(f"Not a folder: {video_dir}")
    iterator = video_dir.rglob("*") if recursive else video_dir.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def top_generator_text(payload: dict[str, Any]) -> str:
    top = payload.get("top_generator")
    if not isinstance(top, dict):
        return "-"
    probability = top.get("probability")
    if isinstance(probability, (int, float)):
        return f"{top.get('name', '-')}:{probability:.4f}"
    return str(top.get("name", "-"))


def print_header() -> None:
    print(
        "\t".join(
            (
                "video",
                "model",
                "label",
                "fake_probability",
                "confidence",
                "top_generator",
                "capture_ms",
                "model_inference_ms",
                "predict_total_ms",
                "wall_total_ms",
            )
        )
    )


def print_row(payload: dict[str, Any]) -> None:
    timings = payload["timings_ms"]
    print(
        "\t".join(
            (
                payload["video"],
                payload["model_id"],
                payload["label"],
                f"{payload['fake_probability']:.4f}",
                f"{payload['confidence']:.4f}",
                top_generator_text(payload),
                f"{timings['capture']:.1f}",
                f"{timings['model_inference']:.1f}",
                f"{timings['predict_total']:.1f}",
                f"{timings['wall_total']:.1f}",
            )
        )
    )


def predict_one_video(
    service: ReferenceModelService,
    video_path: Path,
    model_id: str,
    load_ms: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    capture_started = time.perf_counter()
    frames = capture_evenly_spaced_frames(video_path)
    capture_ms = (time.perf_counter() - capture_started) * 1000.0

    predict_started = time.perf_counter()
    result = service.predict(frames)
    predict_wall_ms = (time.perf_counter() - predict_started) * 1000.0

    return result_payload(
        result=result,
        model_id=model_id,
        video_path=video_path,
        capture_ms=capture_ms,
        load_ms=load_ms,
        predict_wall_ms=predict_wall_ms,
        wall_total_ms=(time.perf_counter() - started) * 1000.0,
    )


def main() -> int:
    args = parse_args()
    video_paths = list_video_files(args.video_dir, recursive=args.recursive)
    if not video_paths:
        raise ValueError(f"No supported videos found in: {args.video_dir}")

    service = ReferenceModelService()
    load_started = time.perf_counter()
    service.load(args.model_id)
    load_ms = (time.perf_counter() - load_started) * 1000.0

    payloads = []
    try:
        if not args.json:
            print_header()
        for video_path in video_paths:
            payload = predict_one_video(
                service=service,
                video_path=video_path,
                model_id=args.model_id,
                load_ms=load_ms,
            )
            payloads.append(payload)
            if not args.json:
                print_row(payload)
    finally:
        service.close()

    if args.json:
        print(
            json.dumps(
                {
                    "model_id": args.model_id,
                    "load_ms": load_ms,
                    "video_count": len(payloads),
                    "results": payloads,
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
