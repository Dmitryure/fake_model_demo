from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .settings import EXPECTED_FRAME_COUNT, MODALITY_FRAME_COUNTS


FACE_CROP_STATUS_DETECTED = "detected"
FACE_CROP_STATUS_FALLBACK = "fallback_full_frame"
HAAR_CASCADE = "haarcascade_frontalface_default.xml"


def validate_frame_count(frame_count: int) -> None:
    if frame_count != EXPECTED_FRAME_COUNT:
        raise ValueError(f"Expected exactly {EXPECTED_FRAME_COUNT} frames, got {frame_count}.")


def select_evenly_spaced(items: Sequence[Any], count: int) -> list[Any]:
    if count <= 0:
        raise ValueError("Selection count must be positive.")
    if len(items) < count:
        raise ValueError(f"Need at least {count} items, got {len(items)}.")
    if len(items) == count:
        return list(items)
    last_index = len(items) - 1
    return [items[round(index * last_index / (count - 1))] for index in range(count)]


def decode_image_bytes(payload: bytes) -> Any:
    import cv2
    import numpy as np

    encoded = np.frombuffer(payload, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Frame is not a valid JPEG/PNG image.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def largest_face_box(boxes: Sequence[Sequence[int]]) -> tuple[int, int, int, int] | None:
    if len(boxes) == 0:
        return None
    x, y, w, h = max(boxes, key=lambda box: int(box[2]) * int(box[3]))
    return int(x), int(y), int(w), int(h)


def median_face_box(boxes: Sequence[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    if not boxes:
        return None
    import numpy as np

    values = np.asarray(boxes, dtype=np.float32)
    x, y, w, h = np.median(values, axis=0).round().astype(np.int64).tolist()
    return int(x), int(y), int(w), int(h)


def enlarge_box(
    box: tuple[int, int, int, int],
    coef: float,
    frame_shape: tuple[int, int, int] | tuple[int, int],
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
    size_w = max(1.0, float(w) * float(coef))
    size_h = max(1.0, float(h) * float(coef))
    center_x = float(x) + float(w) / 2.0
    center_y = float(y) + float(h) / 2.0
    x1 = max(0, round(center_x - size_w / 2.0))
    y1 = max(0, round(center_y - size_h / 2.0))
    x2 = min(frame_w, round(center_x + size_w / 2.0))
    y2 = min(frame_h, round(center_y + size_h / 2.0))
    if x2 <= x1:
        x2 = min(frame_w, x1 + 1)
    if y2 <= y1:
        y2 = min(frame_h, y1 + 1)
    return x1, y1, x2 - x1, y2 - y1


def build_haar_detector() -> Any:
    import cv2

    cascade_path = cv2.data.haarcascades + HAAR_CASCADE
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Could not load OpenCV Haar cascade: {cascade_path}")
    return detector


def detect_largest_face(frame_rgb: Any, detector: Any) -> tuple[int, int, int, int] | None:
    import cv2

    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    boxes = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    return largest_face_box(boxes)


def resolve_face_crop_box(
    frames: Sequence[Any],
    detection_frequency: int = 16,
    large_box_coef: float = 1.5,
) -> tuple[tuple[int, int, int, int] | None, str]:
    if not frames:
        raise ValueError("Cannot crop empty frame sequence.")
    detector = build_haar_detector()
    stride = max(1, int(detection_frequency))
    detected = []
    for frame in frames[::stride]:
        box = detect_largest_face(frame, detector)
        if box is not None:
            detected.append(box)
    median_box = median_face_box(detected)
    if median_box is None:
        return None, FACE_CROP_STATUS_FALLBACK
    return enlarge_box(median_box, large_box_coef, frames[0].shape), FACE_CROP_STATUS_DETECTED


def crop_rgb_frames(
    frames: Sequence[Any],
    box: tuple[int, int, int, int] | None,
) -> list[Any]:
    if box is None:
        return list(frames)
    x, y, w, h = box
    return [frame[y : y + h, x : x + w] for frame in frames]


def preprocess_frames_for_modalities(
    frames: Sequence[Any],
    detection_frequency: int = 16,
    large_box_coef: float = 1.5,
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    validate_frame_count(len(frames))
    crop_box, crop_status = resolve_face_crop_box(
        frames,
        detection_frequency=detection_frequency,
        large_box_coef=large_box_coef,
    )
    cropped = crop_rgb_frames(frames, crop_box)
    by_modality = {
        name: select_evenly_spaced(cropped, frame_count)
        for name, frame_count in MODALITY_FRAME_COUNTS.items()
    }
    return by_modality, {
        "face_crop_status": crop_status,
        "face_crop_box": None if crop_box is None else list(crop_box),
        "input_frame_count": len(frames),
        "modality_frame_counts": dict(MODALITY_FRAME_COUNTS),
    }

