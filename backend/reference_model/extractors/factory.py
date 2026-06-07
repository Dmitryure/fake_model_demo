from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from encoders import build_local_encoders
from encoders.depth import DEFAULT_DEPTH_MODEL_ID
from extractors.base import FeatureExtractor
from extractors.depth import DepthExtractor
from extractors.eye_gaze import build_eye_gaze_extractor
from extractors.face_mesh import build_face_mesh_extractor
from extractors.rgb import RGBExtractor


@dataclass(frozen=True)
class ExtractorFactoryResult:
    extractors: dict[str, FeatureExtractor]
    warnings: tuple[str, ...]


def _require_encoder(encoder: Any, message: str) -> Any:
    if encoder is None:
        raise RuntimeError(message)
    return encoder


def _config_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"`{key}` must be a YAML mapping.")
    return value


def _build_rgb_extractor(config: Mapping[str, Any], encoder_result: Any) -> FeatureExtractor:
    return RGBExtractor(
        _require_encoder(
            encoder_result.rgb_encoder,
            "RGB encoder was not built for the selected modalities.",
        ),
        image_size=int(config.get("image_size", 224)),
    )


def _build_eye_gaze_extractor(config: Mapping[str, Any], encoder_result: Any) -> FeatureExtractor:
    del encoder_result
    return build_eye_gaze_extractor(config)


def _build_face_mesh_extractor(config: Mapping[str, Any], encoder_result: Any) -> FeatureExtractor:
    del encoder_result
    return build_face_mesh_extractor(config)


def _build_depth_extractor(config: Mapping[str, Any], encoder_result: Any) -> FeatureExtractor:
    depth_config = _config_mapping(config, "depth")
    model_id_or_path = depth_config.get("model_id_or_path", DEFAULT_DEPTH_MODEL_ID)
    if not isinstance(model_id_or_path, str) or not model_id_or_path.strip():
        raise ValueError("`depth.model_id_or_path` must be a non-empty string.")
    return DepthExtractor(
        _require_encoder(
            encoder_result.depth_encoder,
            "Depth encoder was not built for the selected modalities.",
        ),
        model_id_or_path=model_id_or_path.strip(),
    )


_EXTRACTOR_BUILDERS: dict[str, Callable[[Mapping[str, Any], Any], FeatureExtractor]] = {
    "rgb": _build_rgb_extractor,
    "eye_gaze": _build_eye_gaze_extractor,
    "face_mesh": _build_face_mesh_extractor,
    "depth": _build_depth_extractor,
}


def _build_extractors_from_encoder_result(
    config: Mapping[str, Any],
    enabled: Sequence[str],
    encoder_result,
) -> dict[str, FeatureExtractor]:
    extractors: dict[str, FeatureExtractor] = {}
    for modality in enabled:
        builder = _EXTRACTOR_BUILDERS.get(modality)
        if builder is not None:
            extractors[modality] = builder(config, encoder_result)
    return extractors


def build_extractors(
    config: Mapping[str, Any],
    modalities: Sequence[str] | None = None,
) -> ExtractorFactoryResult:
    enabled = tuple(modalities or ("rgb", "eye_gaze", "face_mesh", "depth"))
    encoder_result = build_local_encoders(config, modalities=enabled)
    return ExtractorFactoryResult(
        extractors=_build_extractors_from_encoder_result(
            config=config,
            enabled=enabled,
            encoder_result=encoder_result,
        ),
        warnings=encoder_result.warnings,
    )


def build_extractors_from_encoders(
    config: Mapping[str, Any],
    encoder_result,
    modalities: Sequence[str] | None = None,
) -> ExtractorFactoryResult:
    enabled = tuple(modalities or ("rgb", "eye_gaze", "face_mesh", "depth"))
    return ExtractorFactoryResult(
        extractors=_build_extractors_from_encoder_result(
            config=config,
            enabled=enabled,
            encoder_result=encoder_result,
        ),
        warnings=encoder_result.warnings,
    )
