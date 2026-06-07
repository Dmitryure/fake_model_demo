from pathlib import Path

from scripts.batch_predict_video import list_video_files


def test_list_video_files_returns_supported_files(tmp_path: Path) -> None:
    keep = tmp_path / "clip.mp4"
    skip = tmp_path / "notes.txt"
    keep.write_bytes(b"")
    skip.write_text("nope", encoding="utf-8")

    assert list_video_files(tmp_path) == [keep]


def test_list_video_files_can_recurse(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    video = nested / "clip.webm"
    video.write_bytes(b"")

    assert list_video_files(tmp_path, recursive=True) == [video]
    assert list_video_files(tmp_path, recursive=False) == []
