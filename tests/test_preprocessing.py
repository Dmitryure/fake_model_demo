from backend.preprocessing import select_evenly_spaced, validate_frame_count


def test_validate_frame_count_accepts_exact_batch() -> None:
    validate_frame_count(32)


def test_validate_frame_count_rejects_other_sizes() -> None:
    try:
        validate_frame_count(31)
    except ValueError as exc:
        assert "Expected exactly 32 frames" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_select_evenly_spaced_keeps_requested_count() -> None:
    assert select_evenly_spaced(list(range(32)), 16) == [
        0,
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        17,
        19,
        21,
        23,
        25,
        27,
        29,
        31,
    ]
