from src.vision.dataset_split import load_dev_holdout


def test_dev_and_holdout_are_disjoint_and_nonempty():
    dev, holdout = load_dev_holdout()
    assert dev and holdout
    dev_ids = {c["id"] for c in dev}
    holdout_ids = {c["id"] for c in holdout}
    assert dev_ids.isdisjoint(holdout_ids)


def test_all_holdout_cases_are_actually_marked_holdout():
    _, holdout = load_dev_holdout()
    assert all(c["split"] == "holdout" for c in holdout)
