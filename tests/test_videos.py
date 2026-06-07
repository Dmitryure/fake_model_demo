from pathlib import Path

from backend.videos import ensure_video_dir


def test_ensure_video_dir_creates_missing_directory(tmp_path: Path) -> None:
    video_dir = tmp_path / "backend" / "videos"

    assert ensure_video_dir(video_dir) == video_dir
    assert video_dir.is_dir()
