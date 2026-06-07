from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch.nn as nn

from encoders.depth import DEFAULT_DEPTH_FEATURE_DIM, DEFAULT_DEPTH_MODEL_ID, DepthAnythingEncoder
from encoders.rgb import RGBEncoder
from frame_config import resolve_modality_frame_count


@dataclass(frozen=True)
class EncoderFactoryResult:
    depth_encoder: nn.Module | None
    rgb_encoder: nn.Module | None
    warnings: tuple[str, ...]


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"`{key}` must be a YAML mapping.")
    return value


def _require_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int):
        raise ValueError(f"`{key}` must be an integer.")
    return value


def _require_str(config: Mapping[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{key}` must be a non-empty string.")
    return value.strip()


def _optional_path(config: Mapping[str, Any], key: str) -> Path | None:
    value = config.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return Path(stripped)
    raise ValueError(f"`{key}` must be a string path or null.")


def build_local_encoders(
    config: Mapping[str, Any],
    modalities: Sequence[str] | None = None,
) -> EncoderFactoryResult:
    enabled = set(modalities or ("rgb", "depth"))
    rgb_config = _require_mapping(config, "rgb") if "rgb" in enabled else {}
    depth_config = _require_mapping(config, "depth") if "depth" in enabled else {}

    rgb_checkpoint_path = _optional_path(rgb_config, "checkpoint_path")

    rgb_encoder = None
    depth_encoder = None
    if "depth" in enabled:
        depth_encoder = DepthAnythingEncoder(
            model_id_or_path=_require_str(depth_config, "model_id_or_path")
            if "model_id_or_path" in depth_config
            else DEFAULT_DEPTH_MODEL_ID,
            feature_dim=int(depth_config.get("feature_dim", DEFAULT_DEPTH_FEATURE_DIM)),
        )
    if "rgb" in enabled:
        if rgb_checkpoint_path is None:
            raise ValueError("RGB checkpoint_path is required when `rgb` modality is enabled.")
        rgb_encoder = RGBEncoder(
            frames=resolve_modality_frame_count(config, "rgb"),
            image_size=_require_int(config, "image_size"),
            checkpoint_path=rgb_checkpoint_path,
        )
    return EncoderFactoryResult(
        depth_encoder=depth_encoder,
        rgb_encoder=rgb_encoder,
        warnings=(),
    )
