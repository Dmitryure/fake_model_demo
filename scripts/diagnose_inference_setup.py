from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.settings import (  # noqa: E402
    ASSET_ROOT,
    INFERENCE_CONFIG_PATH,
    MODEL_VARIANTS,
    PROJECT_ROOT as SETTINGS_PROJECT_ROOT,
    RUNTIME_ROOT,
    VIDEO_DIR,
    resolve_runtime_asset_paths,
)


OK = "OK"
WARN = "WARN"
FAIL = "FAIL"
PACKAGE_IMPORTS = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "python-multipart": "multipart",
    "PyYAML": "yaml",
    "numpy": "numpy",
    "opencv-python": "cv2",
    "torch": "torch",
    "torchvision": "torchvision",
    "mediapipe": "mediapipe",
    "transformers": "transformers",
}
ENV_NAMES = (
    "FACE_DETECT_DEVICE",
    "FACE_DETECT_ALLOW_CPU",
    "TRANSFORMERS_OFFLINE",
    "HF_HUB_OFFLINE",
    "CUDA_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
)
RUNTIME_FILES = (
    "pipeline.py",
    "encoders/depth.py",
    "encoders/rgb.py",
    "extractors/depth.py",
    "extractors/eye_gaze.py",
    "extractors/face_mesh.py",
    "extractors/mediapipe_face_landmarker.py",
    "task_models/generator_multitask_classifier.py",
)
DEPTH_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
)


@dataclass(frozen=True)
class Check:
    section: str
    name: str
    status: str
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check local files and environment needed for inference without loading model weights.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Optional video path to verify OpenCV can open and read frames.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON.",
    )
    parser.add_argument(
        "--import-packages",
        action="store_true",
        help="Also import non-Torch packages. Slower; useful when package metadata is present but imports fail.",
    )
    return parser.parse_args()


def format_size(size_bytes: int) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    size = float(size_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size_bytes} B"


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def path_check(section: str, name: str, path: Path, kind: str = "file") -> Check:
    if kind == "dir":
        if path.is_dir():
            return Check(section, name, OK, f"{relative(path)} exists")
        return Check(section, name, FAIL, f"{relative(path)} missing directory")
    if path.is_file():
        return Check(section, name, OK, f"{relative(path)} exists ({format_size(path.stat().st_size)})")
    return Check(section, name, FAIL, f"{relative(path)} missing file")


def json_file_check(section: str, name: str, path: Path) -> list[Check]:
    checks = [path_check(section, name, path)]
    if checks[-1].status != OK:
        return checks
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        checks.append(Check(section, f"{name} parse", FAIL, repr(exc)))
        return checks
    if isinstance(data, dict):
        checks.append(Check(section, f"{name} parse", OK, "valid JSON object"))
    else:
        checks.append(Check(section, f"{name} parse", FAIL, f"expected JSON object, got {type(data).__name__}"))
    return checks


def package_check(distribution: str, import_name: str) -> Check:
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return Check("packages", distribution, FAIL, "not installed")
    try:
        module = importlib.import_module(import_name)
    except Exception as exc:
        return Check("packages", distribution, FAIL, f"version={version}; import failed: {exc!r}")
    module_file = getattr(module, "__file__", "-")
    return Check("packages", distribution, OK, f"version={version}; import={import_name}; file={module_file}")


def package_metadata_check(distribution: str) -> Check:
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return Check("packages", distribution, FAIL, "not installed")
    return Check("packages", distribution, OK, f"version={version}")


def torch_checks() -> list[Check]:
    try:
        import torch
    except Exception as exc:
        return [Check("torch", "import", FAIL, repr(exc))]

    checks = [
        Check("torch", "version", OK, str(getattr(torch, "__version__", "-"))),
        Check("torch", "compiled cuda", OK, str(getattr(torch.version, "cuda", None))),
        Check("torch", "has float8_e8m0fnu", OK if hasattr(torch, "float8_e8m0fnu") else WARN, str(hasattr(torch, "float8_e8m0fnu"))),
    ]
    try:
        available = bool(torch.cuda.is_available())
        checks.append(Check("torch", "cuda available", OK if available else WARN, str(available)))
        checks.append(Check("torch", "cuda device count", OK if torch.cuda.device_count() else WARN, str(torch.cuda.device_count())))
        for index in range(torch.cuda.device_count()):
            try:
                name = torch.cuda.get_device_name(index)
            except Exception as exc:
                checks.append(Check("torch", f"cuda device {index}", FAIL, repr(exc)))
            else:
                checks.append(Check("torch", f"cuda device {index}", OK, name))
    except Exception as exc:
        checks.append(Check("torch", "cuda probe", FAIL, repr(exc)))
    return checks


def command_check(name: str, command: list[str]) -> Check:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return Check("system", name, WARN, "command not found")
    except Exception as exc:
        return Check("system", name, WARN, repr(exc))
    output = (result.stdout or result.stderr).strip()
    first_lines = "\n".join(output.splitlines()[:8])
    status = OK if result.returncode == 0 else WARN
    return Check("system", name, status, first_lines or f"exit code {result.returncode}")


def load_yaml_config(path: Path) -> tuple[dict[str, Any] | None, list[Check]]:
    checks = [path_check("config", "inference_config.yaml", path)]
    if checks[-1].status != OK:
        return None, checks
    try:
        import yaml
    except Exception as exc:
        checks.append(Check("config", "PyYAML import", FAIL, repr(exc)))
        return None, checks
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except Exception as exc:
        checks.append(Check("config", "parse", FAIL, repr(exc)))
        return None, checks
    if not isinstance(data, dict):
        checks.append(Check("config", "parse", FAIL, f"expected mapping, got {type(data).__name__}"))
        return None, checks
    checks.append(Check("config", "parse", OK, "valid YAML mapping"))
    return data, checks


def config_reference_checks(config: dict[str, Any]) -> list[Check]:
    checks: list[Check] = []
    resolved = resolve_runtime_asset_paths(config, ASSET_ROOT)
    modalities = resolved.get("modalities")
    if isinstance(modalities, list) and modalities:
        checks.append(Check("config", "modalities", OK, ", ".join(str(item) for item in modalities)))
    else:
        checks.append(Check("config", "modalities", FAIL, "missing or empty `modalities` list"))

    device = str(resolved.get("device", ""))
    checks.append(Check("config", "device", OK if device else FAIL, device or "missing"))
    checks.extend(config_file_refs(resolved))
    return checks


def config_file_refs(config: dict[str, Any]) -> list[Check]:
    checks: list[Check] = []
    checks.extend(config_section_file_check(config, "rgb", "checkpoint_path"))
    checks.extend(config_section_file_check(config, "eye_gaze", "model_path"))
    checks.extend(config_section_path_check(config, "depth", "model_id_or_path"))
    checks.extend(config_section_optional_file_check(config, "fusion", "checkpoint_path"))
    return checks


def config_section_file_check(config: dict[str, Any], section_name: str, key: str) -> list[Check]:
    section = config.get(section_name)
    if not isinstance(section, dict):
        return [Check("config refs", f"{section_name}.{key}", FAIL, f"missing `{section_name}` section")]
    value = section.get(key)
    if not value:
        return [Check("config refs", f"{section_name}.{key}", FAIL, "missing value")]
    return [path_check("config refs", f"{section_name}.{key}", Path(str(value)))]


def config_section_optional_file_check(config: dict[str, Any], section_name: str, key: str) -> list[Check]:
    section = config.get(section_name)
    if not isinstance(section, dict):
        return [Check("config refs", f"{section_name}.{key}", WARN, f"missing `{section_name}` section")]
    value = section.get(key)
    if value is None:
        return [Check("config refs", f"{section_name}.{key}", OK, "not configured")]
    if not value:
        return [Check("config refs", f"{section_name}.{key}", WARN, "empty value")]
    return [path_check("config refs", f"{section_name}.{key}", Path(str(value)))]


def config_section_path_check(config: dict[str, Any], section_name: str, key: str) -> list[Check]:
    checks: list[Check] = []
    section = config.get(section_name)
    if not isinstance(section, dict):
        return [Check("config refs", f"{section_name}.{key}", FAIL, f"missing `{section_name}` section")]
    value = section.get(key)
    if not value:
        return [Check("config refs", f"{section_name}.{key}", FAIL, "missing value")]
    path = Path(str(value))
    checks.append(path_check("config refs", f"{section_name}.{key}", path, kind="dir"))
    if path.is_dir() and path.name == "depth-anything-v2-small-hf":
        for filename in DEPTH_MODEL_FILES:
            checks.append(path_check("config refs", f"depth.{filename}", path / filename))
    return checks


def model_variant_checks() -> list[Check]:
    checks: list[Check] = []
    for model_id, paths in MODEL_VARIANTS.items():
        checks.append(path_check("model weights", f"{model_id} run dir", paths.run_dir, kind="dir"))
        checks.append(path_check("model weights", f"{model_id} best.pt", paths.checkpoint_path))
        checks.extend(json_file_check("model weights", f"{model_id} run_config.json", paths.run_config_path))
    return checks


def runtime_checks() -> list[Check]:
    checks = [
        path_check("runtime", "backend/reference_model", RUNTIME_ROOT, kind="dir"),
        path_check("runtime", "backend/assets", ASSET_ROOT, kind="dir"),
        path_check("runtime", "backend/videos", VIDEO_DIR, kind="dir"),
    ]
    for filename in RUNTIME_FILES:
        checks.append(path_check("runtime", filename, RUNTIME_ROOT / filename))
    return checks


def env_checks() -> list[Check]:
    checks = []
    for name in ENV_NAMES:
        value = os.getenv(name)
        checks.append(Check("environment", name, OK if value else WARN, value or "not set"))
    return checks


def system_checks() -> list[Check]:
    return [
        Check("system", "project root", OK if PROJECT_ROOT == SETTINGS_PROJECT_ROOT else WARN, str(PROJECT_ROOT)),
        Check("system", "cwd", OK, str(Path.cwd())),
        Check("system", "python", OK, sys.executable),
        Check("system", "python version", OK, sys.version.replace("\n", " ")),
        Check("system", "platform", OK, platform.platform()),
        command_check("nvidia-smi", ["nvidia-smi"]),
        command_check("nvcc", ["nvcc", "--version"]),
    ]


def device_gate_checks(config: dict[str, Any] | None) -> list[Check]:
    if config is None:
        return [Check("run gates", "device", FAIL, "cannot evaluate without config")]
    requested = os.getenv("FACE_DETECT_DEVICE")
    allow_cpu = os.getenv("FACE_DETECT_ALLOW_CPU")
    config_device = str(config.get("device", "")).lower()
    if requested:
        return [Check("run gates", "device", OK, f"FACE_DETECT_DEVICE={requested} overrides config device={config_device}")]
    if config_device != "cuda":
        return [Check("run gates", "device", OK, f"config device={config_device}")]
    if allow_cpu == "1":
        return [Check("run gates", "device", OK, "FACE_DETECT_ALLOW_CPU=1 permits CPU fallback")]
    try:
        import torch
        cuda_ok = bool(torch.cuda.is_available())
    except Exception as exc:
        return [Check("run gates", "device", FAIL, f"config requires CUDA; torch probe failed: {exc!r}")]
    if cuda_ok:
        return [Check("run gates", "device", OK, "config requires CUDA and torch sees CUDA")]
    return [Check("run gates", "device", FAIL, "config requires CUDA; torch.cuda.is_available() is False")]


def video_checks(video_path: Path | None) -> list[Check]:
    if video_path is None:
        return [Check("video", "optional video", WARN, "not provided")]
    checks = [path_check("video", "path", video_path)]
    if checks[-1].status != OK:
        return checks
    try:
        import cv2
    except Exception as exc:
        checks.append(Check("video", "cv2 import", FAIL, repr(exc)))
        return checks
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            checks.append(Check("video", "open", FAIL, "cv2.VideoCapture could not open file"))
            return checks
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        checks.append(Check("video", "frame count", OK if frame_count >= 32 else WARN, str(frame_count)))
        ok, frame = capture.read()
        if ok and frame is not None:
            checks.append(Check("video", "first frame", OK, f"shape={getattr(frame, 'shape', None)}"))
        else:
            checks.append(Check("video", "first frame", FAIL, "could not read first frame"))
    finally:
        capture.release()
    return checks


def package_checks(import_packages: bool) -> list[Check]:
    if import_packages:
        return [
            package_check(distribution, import_name)
            for distribution, import_name in PACKAGE_IMPORTS.items()
        ]
    return [package_metadata_check(distribution) for distribution in PACKAGE_IMPORTS]


def collect_checks(video_path: Path | None, import_packages: bool) -> list[Check]:
    config, config_checks = load_yaml_config(INFERENCE_CONFIG_PATH)
    checks: list[Check] = []
    checks.extend(system_checks())
    checks.extend(env_checks())
    checks.extend(package_checks(import_packages))
    checks.extend(torch_checks())
    checks.extend(runtime_checks())
    checks.extend(config_checks)
    if config is not None:
        checks.extend(config_reference_checks(config))
    checks.extend(model_variant_checks())
    checks.extend(device_gate_checks(config))
    checks.extend(video_checks(video_path))
    return checks


def check_to_dict(check: Check) -> dict[str, str]:
    return {
        "section": check.section,
        "name": check.name,
        "status": check.status,
        "detail": check.detail,
    }


def print_text(checks: list[Check]) -> None:
    max_section = max(len(check.section) for check in checks)
    max_name = max(len(check.name) for check in checks)
    for check in checks:
        print(
            f"[{check.status:<4}] {check.section:<{max_section}}  "
            f"{check.name:<{max_name}}  {check.detail}"
        )
    fail_count = sum(1 for check in checks if check.status == FAIL)
    warn_count = sum(1 for check in checks if check.status == WARN)
    print(f"\nSummary: {fail_count} fail, {warn_count} warn, {len(checks)} checks")


def main() -> int:
    args = parse_args()
    checks = collect_checks(args.video, import_packages=args.import_packages)
    if args.json:
        print(json.dumps([check_to_dict(check) for check in checks], indent=2))
    else:
        print_text(checks)
    return 1 if any(check.status == FAIL for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
