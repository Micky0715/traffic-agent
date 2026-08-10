from src.vision.bad_case_report import collect_bad_cases


def _fake_report() -> dict:
    return {
        "experiments": {
            "A": {"results": [
                {"id": "X1", "image": "x1.png", "category": "normal", "score": 1.0},
                {"id": "X2", "image": "x2.png", "category": "blur", "score": 0.5},
            ]},
            "C": {"results": [
                {"id": "X1", "image": "x1.png", "category": "normal", "score": 1.0},
                {"id": "X2", "image": "x2.png", "category": "blur", "score": 1.0},
            ]},
        }
    }


def test_perfect_case_is_not_collected():
    cases = collect_bad_cases(_fake_report())
    ids = {c.case_id for c in cases}
    assert "X1" not in ids


def test_case_failing_on_only_one_experiment_is_still_collected():
    cases = collect_bad_cases(_fake_report())
    x2 = next(c for c in cases if c.case_id == "X2")
    assert x2.experiment_scores == {"A": 0.5, "C": 1.0}


def test_collected_case_keeps_per_experiment_scores_side_by_side():
    """A case failing on the mock OCR path but passing on real VLM is a
    fundamentally different bad case than one failing everywhere — the
    per-experiment breakdown must not be collapsed into one number."""
    cases = collect_bad_cases(_fake_report())
    x2 = next(c for c in cases if c.case_id == "X2")
    assert x2.experiment_scores["A"] != x2.experiment_scores["C"]
