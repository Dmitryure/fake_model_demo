from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "backend" / "assets" / "models" / "face_landmarker_v2_with_blendshapes.task"
)
CASES = (
    "default_path_blendshapes",
    "cpu_path_blendshapes",
    "cpu_path_no_blendshapes",
    "cpu_buffer_blendshapes",
    "create_from_model_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe MediaPipe FaceLandmarker init variants with timeouts.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to face_landmarker_v2_with_blendshapes.task.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="Seconds to allow each child probe.",
    )
    parser.add_argument(
        "--case",
        choices=CASES,
        help="Internal: run one probe case.",
    )
    return parser.parse_args()


def env_for_child() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("GLOG_logtostderr", "1")
    env.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
    return env


def run_all(args: argparse.Namespace) -> int:
    print(f"python={sys.executable}", flush=True)
    print(f"model_path={args.model_path}", flush=True)
    print(f"model_exists={args.model_path.exists()}", flush=True)
    print(f"timeout={args.timeout}s", flush=True)

    failures = 0
    for case in CASES:
        started = time.perf_counter()
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--model-path",
            str(args.model_path),
            "--case",
            case,
        ]
        print(f"\n>>> CASE {case}", flush=True)
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                check=False,
                env=env_for_child(),
                text=True,
                timeout=args.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            failures += 1
            elapsed = time.perf_counter() - started
            print(f"TIMEOUT after {elapsed:.1f}s", flush=True)
            if exc.stdout:
                print(exc.stdout, end="")
            if exc.stderr:
                print(exc.stderr, end="", file=sys.stderr)
            continue

        elapsed = time.perf_counter() - started
        status = "OK" if result.returncode == 0 else "FAIL"
        if result.returncode != 0:
            failures += 1
        print(f"{status} exit={result.returncode} elapsed={elapsed:.1f}s", flush=True)
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)

    print(f"\nSummary: {failures} failing cases out of {len(CASES)}", flush=True)
    return 1 if failures else 0


def run_case(case: str, model_path: Path) -> int:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("GLOG_logtostderr", "1")
    os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    print(f"mediapipe={getattr(mp, '__version__', '-')}", flush=True)
    print(f"case={case}", flush=True)

    if case == "create_from_model_path":
        landmarker = vision.FaceLandmarker.create_from_model_path(str(model_path))
        landmarker.close()
        return 0

    base_options_kwargs: dict[str, Any] = {}
    if case == "cpu_buffer_blendshapes":
        base_options_kwargs["model_asset_buffer"] = model_path.read_bytes()
    else:
        base_options_kwargs["model_asset_path"] = str(model_path)

    if case.startswith("cpu_"):
        base_options_kwargs["delegate"] = python.BaseOptions.Delegate.CPU

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(**base_options_kwargs),
        running_mode=vision.RunningMode.IMAGE,
        output_face_blendshapes=case != "cpu_path_no_blendshapes",
        output_facial_transformation_matrixes=False,
        min_face_detection_confidence=0.7,
        num_faces=2,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    landmarker.close()
    return 0


def main() -> int:
    args = parse_args()
    if args.case:
        return run_case(args.case, args.model_path)
    return run_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
