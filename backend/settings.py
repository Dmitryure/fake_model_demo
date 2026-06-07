from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PROJECT_ROOT / "backend" / "reference_model"
ASSET_ROOT = PROJECT_ROOT / "backend" / "assets"
MODEL_WEIGHTS_DIR = PROJECT_ROOT / "backend" / "model_weights"
INFERENCE_CONFIG_PATH = PROJECT_ROOT / "backend" / "inference_config.yaml"
DEFAULT_MODEL_ID = "canonical_best"
MODEL_DOC_PATH = PROJECT_ROOT / "docs" / "model_notes.md"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
VIDEO_DIR = PROJECT_ROOT / "backend" / "videos"
EXPECTED_FRAME_COUNT = 32
BINARY_THRESHOLD = 0.5
MODALITY_FRAME_COUNTS = {
    "rgb": 16,
    "eye_gaze": 32,
    "face_mesh": 32,
    "depth": 32,
}


@dataclass(frozen=True)
class ModelPaths:
    model_id: str = DEFAULT_MODEL_ID
    display_name: str = "Canonical best"
    runtime_root: Path = RUNTIME_ROOT
    asset_root: Path = ASSET_ROOT
    run_dir: Path = MODEL_WEIGHTS_DIR / "canonical_best"
    checkpoint_path: Path = MODEL_WEIGHTS_DIR / "canonical_best" / "best.pt"
    config_path: Path = INFERENCE_CONFIG_PATH
    run_config_path: Path = MODEL_WEIGHTS_DIR / "canonical_best" / "run_config.json"
    model_doc_path: Path = MODEL_DOC_PATH


MODEL_VARIANTS: dict[str, ModelPaths] = {
    "canonical_best": ModelPaths(),
    "ffpp_celebdf": ModelPaths(
        model_id="ffpp_celebdf",
        display_name="FF++ + CelebDF",
        run_dir=MODEL_WEIGHTS_DIR / "ffpp_celebdf",
        checkpoint_path=MODEL_WEIGHTS_DIR / "ffpp_celebdf" / "best.pt",
        run_config_path=MODEL_WEIGHTS_DIR / "ffpp_celebdf" / "run_config.json",
    ),
}


def available_model_variants() -> list[dict[str, str]]:
    return [
        {
            "id": model_id,
            "name": paths.display_name,
            "checkpoint": str(paths.checkpoint_path),
        }
        for model_id, paths in MODEL_VARIANTS.items()
    ]


def model_paths_for_id(model_id: str | None) -> ModelPaths:
    key = model_id or DEFAULT_MODEL_ID
    try:
        return MODEL_VARIANTS[key]
    except KeyError:
        available = ", ".join(sorted(MODEL_VARIANTS))
        raise ValueError(f"Unknown model_id {key!r}. Available: {available}") from None


def load_run_config(paths: ModelPaths = ModelPaths()) -> dict[str, Any]:
    with paths.run_config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Run config must be JSON object: {paths.run_config_path}")
    return data


def generator_names_from_run_config(run_config: dict[str, Any]) -> tuple[str, ...]:
    target = run_config.get("target")
    if not isinstance(target, dict):
        raise ValueError("Run config missing `target` object.")
    names = target.get("generator_names")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("Run config target missing `generator_names` list.")
    return tuple(names)


def _resolve_asset_path(value: Any, asset_root: Path) -> Any:
    if value is None:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return str(path)
    return str(asset_root / path)


def resolve_runtime_asset_paths(config: dict[str, Any], asset_root: Path) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    for section_name, keys in {
        "rgb": ("checkpoint_path",),
        "eye_gaze": ("model_path",),
        "depth": ("model_id_or_path",),
        "fusion": ("checkpoint_path",),
    }.items():
        section = resolved.get(section_name)
        if not isinstance(section, dict):
            continue
        for key in keys:
            if key in section and section[key] is not None:
                section[key] = _resolve_asset_path(section[key], asset_root)
    return resolved


def choose_device(config: dict[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    requested = os.getenv("FACE_DETECT_DEVICE")
    if requested:
        resolved["device"] = requested
        return resolved
    if str(resolved.get("device", "")).lower() == "cuda" and os.getenv("FACE_DETECT_ALLOW_CPU") != "1":
        try:
            import torch
        except ImportError:
            raise RuntimeError("Config requires CUDA, but torch is not installed.") from None
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Config requires CUDA, but torch cannot see a CUDA device. "
                "Check NVIDIA driver/WSL GPU passthrough with `nvidia-smi`."
            )
    if str(resolved.get("device", "")).lower() == "cuda":
        return resolved
    if os.getenv("FACE_DETECT_ALLOW_CPU") == "1":
        resolved["device"] = "cpu"
    return resolved
