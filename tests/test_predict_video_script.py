from scripts.predict_video import evenly_spaced_indices


def test_evenly_spaced_indices_selects_requested_count() -> None:
    assert evenly_spaced_indices(10, 4) == [0, 3, 6, 9]


def test_evenly_spaced_indices_rejects_short_video() -> None:
    try:
        evenly_spaced_indices(3, 4)
    except ValueError as exc:
        assert "Need at least 4 video frames" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
