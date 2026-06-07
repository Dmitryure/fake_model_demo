from __future__ import annotations

import sys
import time
import logging
from dataclasses import dataclass
from typing import Any

from .preprocessing import preprocess_frames_for_modalities
from .settings import (
    BINARY_THRESHOLD,
    ModelPaths,
    available_model_variants,
    choose_device,
    generator_names_from_run_config,
    load_run_config,
    model_paths_for_id,
    resolve_runtime_asset_paths,
)


logger = logging.getLogger("best_run_detector")


@dataclass(frozen=True)
class PredictionResult:
    label: str
    fake_probability: float
    real_probability: float
    confidence: float
    confidence_label: str
    threshold: float
    generator_debug: dict[str, Any]
    preprocessing: dict[str, Any]
    timings_ms: dict[str, float]


def confidence_label(confidence: float) -> str:
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.65:
        return "medium"
    return "low"


def _ensure_runtime_import_path(paths: ModelPaths) -> None:
    runtime_root = str(paths.runtime_root)
    if runtime_root not in sys.path:
        sys.path.insert(0, runtime_root)


class ReferenceModelService:
    def __init__(self, paths: ModelPaths | None = None) -> None:
        self.paths = paths or ModelPaths()
        self.model_id = self.paths.model_id
        self.config: dict[str, Any] | None = None
        self.generator_names: tuple[str, ...] = ()
        self.device: Any | None = None
        self.model: Any | None = None
        self.load_warnings: tuple[str, ...] = ()
        self.loaded_at: float | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self, model_id: str | None = None) -> None:
        paths = model_paths_for_id(model_id) if model_id is not None else self.paths
        if self.model is not None and paths.model_id == self.model_id:
            return
        self.close()
        self.paths = paths
        self.model_id = paths.model_id
        _ensure_runtime_import_path(self.paths)
        logger.info("load: importing torch and runtime modules")
        import torch
        from pipeline import build_fusion_pipeline, load_pipeline_yaml
        from task_models.generator_multitask_classifier import (
            build_generator_multitask_classifier,
        )

        logger.info("load: reading config path=%s", self.paths.config_path)
        config = load_pipeline_yaml(self.paths.config_path)
        config = resolve_runtime_asset_paths(config, self.paths.asset_root)
        config = choose_device(config)
        logger.info("load: resolved device=%s modalities=%s", config.get("device"), config.get("modalities"))
        run_config = load_run_config(self.paths)
        self.generator_names = generator_names_from_run_config(run_config)

        logger.info("load: building fusion pipeline")
        build_result = build_fusion_pipeline(config=config, modalities=tuple(config["modalities"]))
        logger.info("load: building classifier head")
        model = build_generator_multitask_classifier(
            build_result.pipeline,
            dim=int(config["dim"]),
            num_generators=len(self.generator_names),
            head_config=config.get("head", {}),
        ).to(build_result.device)
        logger.info("load: loading checkpoint path=%s", self.paths.checkpoint_path)
        state = torch.load(
            self.paths.checkpoint_path,
            map_location=build_result.device,
            weights_only=False,
        )
        logger.info("load: applying checkpoint")
        model.load_state_dict(state)
        model.eval()

        self.config = config
        self.device = build_result.device
        self.model = model
        self.load_warnings = tuple(str(item) for item in build_result.warnings)
        self.loaded_at = time.time()
        logger.info("load: complete device=%s warnings=%s", self.device, self.load_warnings)

    def metadata(self) -> dict[str, Any]:
        return {
            "loaded": self.is_loaded,
            "model_id": self.model_id,
            "model_name": self.paths.display_name,
            "available_models": available_model_variants(),
            "loaded_at": self.loaded_at,
            "device": None if self.device is None else str(self.device),
            "modalities": [] if self.config is None else list(self.config.get("modalities", [])),
            "generator_names": list(self.generator_names),
            "paths": {
                "runtime_root": str(self.paths.runtime_root),
                "asset_root": str(self.paths.asset_root),
                "run_dir": str(self.paths.run_dir),
                "checkpoint": str(self.paths.checkpoint_path),
                "config": str(self.paths.config_path),
                "model_docs": str(self.paths.model_doc_path),
            },
            "warnings": list(self.load_warnings),
        }

    def predict(self, frames: list[Any]) -> PredictionResult:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        import torch

        started = time.perf_counter()
        frames_by_modality, preprocessing_debug = preprocess_frames_for_modalities(frames)
        preprocess_done = time.perf_counter()
        batch = {
            "video_rgb_frames_by_modality": frames_by_modality,
            "video_rgb_frames": frames_by_modality["eye_gaze"],
        }
        with torch.no_grad():
            output = self.model(batch)
        inference_done = time.perf_counter()

        fake_probability = float(output.binary_probabilities.detach().cpu().view(-1)[0])
        real_probability = 1.0 - fake_probability
        label = "Fake" if fake_probability >= BINARY_THRESHOLD else "Real"
        confidence = max(fake_probability, real_probability)
        generator_probabilities = output.generator_probabilities.detach().cpu().view(-1).tolist()
        generator_rows = [
            {"name": name, "probability": float(probability)}
            for name, probability in zip(self.generator_names, generator_probabilities, strict=True)
        ]
        generator_rows.sort(key=lambda row: row["probability"], reverse=True)

        diagnostics = {}
        for key, value in output.diagnostics.items():
            if hasattr(value, "detach"):
                diagnostics[key] = value.detach().cpu().tolist()

        return PredictionResult(
            label=label,
            fake_probability=fake_probability,
            real_probability=real_probability,
            confidence=confidence,
            confidence_label=confidence_label(confidence),
            threshold=BINARY_THRESHOLD,
            generator_debug={
                "top_generator": generator_rows[0] if generator_rows else None,
                "generator_probabilities": generator_rows,
                "diagnostics": diagnostics,
            },
            preprocessing=preprocessing_debug,
            timings_ms={
                "preprocess": (preprocess_done - started) * 1000.0,
                "inference": (inference_done - preprocess_done) * 1000.0,
                "total": (inference_done - started) * 1000.0,
            },
        )

    def close(self) -> None:
        if self.model is None:
            return
        pipeline = getattr(self.model, "pipeline", None)
        close = getattr(pipeline, "close", None)
        if callable(close):
            close()
        self.config = None
        self.generator_names = ()
        self.device = None
        self.model = None
        self.load_warnings = ()
        self.loaded_at = None
