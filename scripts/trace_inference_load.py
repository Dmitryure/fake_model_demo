from __future__ import annotations

import argparse
import faulthandler
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.settings import (  # noqa: E402
    DEFAULT_MODEL_ID,
    MODEL_VARIANTS,
    ModelPaths,
    choose_device,
    generator_names_from_run_config,
    load_run_config,
    model_paths_for_id,
    resolve_runtime_asset_paths,
)


ORIGINAL_SOCKET_CONNECT = socket.socket.connect
ORIGINAL_CREATE_CONNECTION = socket.create_connection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace model-load stages and dump Python stacks if inference startup hangs.",
    )
    parser.add_argument(
        "--model-id",
        choices=sorted(MODEL_VARIANTS),
        default=DEFAULT_MODEL_ID,
        help="Checkpoint variant to load.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        help="Override FACE_DETECT_DEVICE for this trace.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=("rgb", "eye_gaze", "face_mesh", "depth"),
        help="Override configured modalities to isolate a hanging modality.",
    )
    parser.add_argument(
        "--dump-after",
        type=int,
        default=20,
        help="Seconds before repeated traceback dumps while a stage is stuck.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Set TRANSFORMERS_OFFLINE=1 and HF_HUB_OFFLINE=1 before loading.",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Do not log socket connection attempts.",
    )
    parser.add_argument(
        "--stop-after",
        choices=(
            "imports",
            "config",
            "encoders",
            "extractors",
            "fusion",
            "registry",
            "classifier",
            "checkpoint",
            "complete",
        ),
        default="complete",
        help="Stop after a stage. Useful with --modalities to isolate startup hangs.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def format_elapsed(started: float) -> str:
    return f"{time.perf_counter() - started:.3f}s"


def begin_stage(name: str) -> float:
    log(f"\n>>> START {name}")
    return time.perf_counter()


def end_stage(name: str, started: float) -> None:
    log(f"<<< END {name} ({format_elapsed(started)})")


def fail_stage(name: str, started: float, exc: BaseException) -> None:
    log(f"xxx FAIL {name} ({format_elapsed(started)}): {exc!r}")


def traced_socket_connect(sock: socket.socket, address: Any) -> Any:
    log(f"!!! NETWORK socket.connect address={address!r}")
    traceback.print_stack(limit=8)
    return ORIGINAL_SOCKET_CONNECT(sock, address)


def traced_create_connection(
    address: Any,
    timeout: float | object = socket._GLOBAL_DEFAULT_TIMEOUT,
    source_address: tuple[str, int] | None = None,
    all_errors: bool = False,
) -> socket.socket:
    log(f"!!! NETWORK socket.create_connection address={address!r}")
    traceback.print_stack(limit=8)
    return ORIGINAL_CREATE_CONNECTION(
        address,
        timeout=timeout,
        source_address=source_address,
        all_errors=all_errors,
    )


def install_network_trace() -> None:
    socket.socket.connect = traced_socket_connect
    socket.create_connection = traced_create_connection


def configure_environment(args: argparse.Namespace) -> None:
    if args.device:
        os.environ["FACE_DETECT_DEVICE"] = args.device
    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"


def log_environment() -> None:
    log(f"python={sys.executable}")
    for name in (
        "FACE_DETECT_DEVICE",
        "FACE_DETECT_ALLOW_CPU",
        "TRANSFORMERS_OFFLINE",
        "HF_HUB_OFFLINE",
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
    ):
        log(f"env {name}={os.getenv(name)}")


def ensure_runtime_import_path(paths: ModelPaths) -> None:
    runtime_root = str(paths.runtime_root)
    if runtime_root not in sys.path:
        sys.path.insert(0, runtime_root)


def stage_import_torch_and_runtime(paths: ModelPaths) -> dict[str, Any]:
    ensure_runtime_import_path(paths)
    import torch
    from branches.compression import validate_branch_token_config
    from encoders import build_local_encoders
    from extractors import build_extractors_from_encoders
    from pipeline import (
        FIXED_SLOT_MODALITIES,
        build_fusion_from_config,
        load_fusion_checkpoint,
        load_pipeline_yaml,
        resolve_model_device,
    )
    from registry import build_registry
    from task_models.generator_multitask_classifier import (
        build_generator_multitask_classifier,
    )

    log(f"torch={torch.__version__} cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()}")
    return {
        "torch": torch,
        "build_fusion_from_config": build_fusion_from_config,
        "build_generator_multitask_classifier": build_generator_multitask_classifier,
        "build_local_encoders": build_local_encoders,
        "build_extractors_from_encoders": build_extractors_from_encoders,
        "build_registry": build_registry,
        "load_fusion_checkpoint": load_fusion_checkpoint,
        "load_pipeline_yaml": load_pipeline_yaml,
        "resolve_model_device": resolve_model_device,
        "validate_branch_token_config": validate_branch_token_config,
        "fixed_slot_modalities": FIXED_SLOT_MODALITIES,
    }


def load_config(paths: ModelPaths, imports: dict[str, Any], modalities: list[str] | None) -> dict[str, Any]:
    config = imports["load_pipeline_yaml"](paths.config_path)
    config = resolve_runtime_asset_paths(config, paths.asset_root)
    config = choose_device(config)
    if modalities:
        config["modalities"] = modalities
    log(f"config device={config.get('device')} modalities={config.get('modalities')}")
    return config


def validate_config(config: dict[str, Any], imports: dict[str, Any]) -> None:
    fusion_config = config["fusion"]
    imports["validate_branch_token_config"](
        config,
        modalities=imports["fixed_slot_modalities"],
        fusion_max_time_steps=int(fusion_config["max_time_steps"]),
    )


def build_encoders(config: dict[str, Any], imports: dict[str, Any]) -> Any:
    return imports["build_local_encoders"](
        config,
        modalities=tuple(config["modalities"]),
    )


def build_extractors(config: dict[str, Any], imports: dict[str, Any], encoder_result: Any) -> Any:
    return imports["build_extractors_from_encoders"](
        config=config,
        encoder_result=encoder_result,
        modalities=tuple(config["modalities"]),
    )


def build_fusion(config: dict[str, Any], imports: dict[str, Any]) -> Any:
    return imports["build_fusion_from_config"](config)


def maybe_load_fusion_checkpoint(config: dict[str, Any], imports: dict[str, Any], fusion_module: Any) -> None:
    checkpoint_path = config.get("fusion", {}).get("checkpoint_path")
    path = None if checkpoint_path is None else Path(str(checkpoint_path))
    loaded = imports["load_fusion_checkpoint"](fusion_module=fusion_module, checkpoint_path=path)
    log(f"fusion checkpoint loaded={loaded}")


def build_registry(config: dict[str, Any], imports: dict[str, Any]) -> Any:
    return imports["build_registry"](dim=int(config["dim"]), config=config)


def build_classifier(
    config: dict[str, Any],
    imports: dict[str, Any],
    pipeline: Any,
    generator_count: int,
) -> Any:
    return imports["build_generator_multitask_classifier"](
        pipeline,
        dim=int(config["dim"]),
        num_generators=generator_count,
        head_config=config.get("head", {}),
    )


def load_model_state(paths: ModelPaths, imports: dict[str, Any], device: Any) -> Any:
    return imports["torch"].load(
        paths.checkpoint_path,
        map_location=device,
        weights_only=False,
    )


def run_stage(name: str, func: Any, *args: Any) -> Any:
    started = begin_stage(name)
    try:
        result = func(*args)
    except BaseException as exc:
        fail_stage(name, started, exc)
        raise
    end_stage(name, started)
    return result


def stop_after(args: argparse.Namespace, stage: str, started: float) -> bool:
    if args.stop_after != stage:
        return False
    faulthandler.cancel_dump_traceback_later()
    log(f"\nTRACE STOPPED after={stage} total={format_elapsed(started)}")
    return True


def main() -> int:
    args = parse_args()
    configure_environment(args)
    if not args.allow_network:
        install_network_trace()
    faulthandler.enable()
    faulthandler.dump_traceback_later(args.dump_after, repeat=True)

    started = time.perf_counter()
    paths = model_paths_for_id(args.model_id)
    log_environment()
    log(f"model_id={paths.model_id} checkpoint={paths.checkpoint_path}")

    imports = run_stage("import torch and runtime modules", stage_import_torch_and_runtime, paths)
    if stop_after(args, "imports", started):
        return 0
    config = run_stage("load and resolve config", load_config, paths, imports, args.modalities)
    if stop_after(args, "config", started):
        return 0
    run_config = run_stage("load run_config.json", load_run_config, paths)
    generator_names = run_stage("parse generator names", generator_names_from_run_config, run_config)
    run_stage("validate branch token config", validate_config, config, imports)
    device = run_stage("resolve model device", imports["resolve_model_device"], config)
    encoder_result = run_stage("build local encoders", build_encoders, config, imports)
    if stop_after(args, "encoders", started):
        return 0
    extractors_result = run_stage("build extractors", build_extractors, config, imports, encoder_result)
    if stop_after(args, "extractors", started):
        return 0
    fusion_module = run_stage("build fusion module", build_fusion, config, imports)
    run_stage("load optional fusion checkpoint", maybe_load_fusion_checkpoint, config, imports, fusion_module)
    if stop_after(args, "fusion", started):
        return 0
    registry = run_stage("build registry", build_registry, config, imports)
    if stop_after(args, "registry", started):
        return 0

    from pipeline import ClipFusionPipeline

    pipeline = ClipFusionPipeline(
        registry=registry,
        fusion_module=fusion_module,
        enabled_modalities=tuple(config["modalities"]),
        extractors=extractors_result.extractors,
        encoder_modules=imports["torch"].nn.ModuleDict({
            name: module
            for name, module in {
                "depth": encoder_result.depth_encoder,
                "rgb": encoder_result.rgb_encoder,
            }.items()
            if module is not None
        }),
    )
    model = run_stage("build classifier", build_classifier, config, imports, pipeline, len(generator_names))
    run_stage("move classifier to device", model.to, device)
    if stop_after(args, "classifier", started):
        return 0
    state = run_stage("torch.load classifier checkpoint", load_model_state, paths, imports, device)
    if stop_after(args, "checkpoint", started):
        return 0
    run_stage("apply classifier checkpoint", model.load_state_dict, state)
    run_stage("set eval mode", model.eval)

    faulthandler.cancel_dump_traceback_later()
    log(f"\nTRACE COMPLETE total={format_elapsed(started)} device={device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
