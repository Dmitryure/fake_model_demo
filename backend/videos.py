from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote


VIDEO_EXTENSIONS = frozenset({".mp4", ".webm", ".mov", ".m4v", ".ogg"})
VIDEO_CHUNK_SIZE = 1024 * 1024
NO_STORE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


@dataclass(frozen=True)
class VideoRange:
    start: int
    end: int
    status_code: int

    @property
    def content_length(self) -> int:
        return self.end - self.start + 1


def is_supported_video(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def ensure_video_dir(video_dir: Path) -> Path:
    video_dir.mkdir(parents=True, exist_ok=True)
    return video_dir


def list_video_files(
    video_dir: Path,
    route_prefix: str = "/v1/video_stream",
) -> list[dict[str, Any]]:
    if not video_dir.exists():
        return []
    rows = []
    for path in sorted(video_dir.iterdir(), key=lambda item: item.name.lower()):
        if not is_supported_video(path):
            continue
        rows.append(
            {
                "name": path.name,
                "url": f"{route_prefix.rstrip('/')}/{quote(path.name, safe='')}",
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def resolve_video_path(video_dir: Path, filename: str) -> Path:
    root = video_dir.resolve()
    path = (root / filename).resolve()
    if path.parent != root or not is_supported_video(path):
        raise ValueError("Video not found.")
    return path


def parse_range_header(range_header: str | None, file_size: int) -> VideoRange:
    if file_size <= 0:
        raise ValueError("Video file is empty.")
    if not range_header:
        return VideoRange(start=0, end=file_size - 1, status_code=200)
    if not range_header.startswith("bytes=") or "," in range_header:
        raise ValueError("Unsupported Range header.")

    start_text, separator, end_text = range_header.removeprefix("bytes=").partition("-")
    if separator != "-":
        raise ValueError("Invalid Range header.")

    if start_text == "":
        suffix_length = int(end_text)
        if suffix_length <= 0:
            raise ValueError("Invalid Range suffix.")
        start = max(file_size - suffix_length, 0)
        end = file_size - 1
    else:
        start = int(start_text)
        end = int(end_text) if end_text else file_size - 1

    if start < 0 or end < start or start >= file_size:
        raise ValueError("Requested range is not satisfiable.")
    return VideoRange(start=start, end=min(end, file_size - 1), status_code=206)


def iter_video_bytes(
    path: Path,
    start: int,
    end: int,
    chunk_size: int = VIDEO_CHUNK_SIZE,
) -> Iterator[bytes]:
    remaining = end - start + 1
    with path.open("rb") as handle:
        handle.seek(start)
        while remaining > 0:
            chunk = handle.read(min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def video_media_type(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def video_stream_headers(video_range: VideoRange, file_size: int) -> dict[str, str]:
    headers = {
        **NO_STORE_HEADERS,
        "Accept-Ranges": "bytes",
        "Content-Length": str(video_range.content_length),
    }
    if video_range.status_code == 206:
        headers["Content-Range"] = f"bytes {video_range.start}-{video_range.end}/{file_size}"
    return headers
