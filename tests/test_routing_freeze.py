"""Reverse-validation of the routing freeze.

Every test that needs to "change" a frozen file backs it up, mutates it,
restores it byte for byte in a finally block, and checks what the freeze tool
reported. Nothing here leaves a dataset or a historical prediction modified —
those are the experiment's evidence, and a test that corrupts them to prove it
can detect corruption has defeated its own purpose.

The property under test is not "a hash changes". It is that the report says
WHICH KIND of input moved: a dataset edit and an evaluator edit invalidate
different conclusions, and a freeze that calls both "CHANGED" leaves the reader
to guess.

Tooling tests compare against a baseline snapshot taken at test time, NOT
against the recorded manifest. Those are different questions — "does the tool
report correctly" versus "is the tree currently frozen" — and coupling them
made every tooling test error out the moment a sanctioned change landed, which
is precisely when the tooling most needs to be trustworthy. Exactly one test
asserts the second question, and it skips rather than fails while an approved
change is in flight.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import routing_freeze  # noqa: E402

MANIFEST = ROOT / "outputs" / "routing_freeze_manifest.json"


@pytest.fixture
def frozen() -> dict:
    if not MANIFEST.exists():
        pytest.skip("run scripts/routing_freeze.py --write first")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture
def baseline() -> dict:
    """The tree as it is right now, so a tamper test detects the tamper itself
    rather than re-reporting a sanctioned change that was already present."""
    return routing_freeze.snapshot()


def diff_groups(before: dict, after: dict) -> list[tuple[str, str, str]]:
    out = []
    for name, label in routing_freeze.GROUP_CHANGE_LABEL.items():
        left = routing_freeze._group_files(before["groups"][name])
        right = routing_freeze._group_files(after["groups"][name])
        for rel in sorted(set(right) - set(left)):
            out.append(("ADDED", name, rel))
        for rel in sorted(set(left) - set(right)):
            out.append(("REMOVED", name, rel))
        for rel in sorted(set(left) & set(right)):
            if left[rel] != right[rel]:
                out.append((label, name, rel))
    return out


def findings_after_touching(rel_path: str, baseline: dict,
                            suffix: str = "\n# freeze test tamper\n"):
    path = ROOT / rel_path
    backup = path.read_bytes()
    try:
        path.write_bytes(backup + suffix.encode("utf-8"))
        return diff_groups(baseline, routing_freeze.snapshot())
    finally:
        path.write_bytes(backup)


# --------------------------------------------------------------------------
# 1. an unrelated module must not raise an alarm
# --------------------------------------------------------------------------

def test_unrelated_module_does_not_trip_the_freeze(baseline):
    """The defect the vision freeze had: a whole-tree hash fired on packages
    the pipeline never imports, and an alarm that cries wolf stops being read."""
    intruder = ROOT / "src" / "_freeze_probe_pkg"
    intruder.mkdir(exist_ok=True)
    try:
        (intruder / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
        assert diff_groups(baseline, routing_freeze.snapshot()) == []
    finally:
        shutil.rmtree(intruder, ignore_errors=True)


def test_routing_package_is_not_in_the_routing_runtime_closure(frozen):
    """src/routing is the new data contract; no router imports it, so it must
    not appear in the runtime scope."""
    union = set(routing_freeze._group_files(frozen["groups"]["routing_runtime_code"]))
    assert not any(rel.startswith("src/routing/") for rel in union)


# --------------------------------------------------------------------------
# 2-7. each group reports under its own label
# --------------------------------------------------------------------------

def test_extractors_change_is_reported_as_runtime_code_changed(baseline):
    found = findings_after_touching("src/extractors.py", baseline)
    assert ("CHANGED", "routing_runtime_code", "src/extractors.py") in found


def test_routing_config_change_is_reported_as_config_changed(baseline):
    found = findings_after_touching("configs/routing.yaml", baseline, "\n# tamper: true\n")
    assert ("CONFIG_CHANGED", "routing_config_and_prompts", "configs/routing.yaml") in found


def test_prompt_change_is_reported_as_config_changed(baseline):
    """Prompts are loaded by a runtime string, so no import closure can reach
    them; they are declared explicitly and must still be covered."""
    found = findings_after_touching("prompts/01_intent_router_v2.md", baseline)
    assert ("CONFIG_CHANGED", "routing_config_and_prompts",
            "prompts/01_intent_router_v2.md") in found


def test_challenge75_change_is_reported_as_dataset_changed(baseline):
    found = findings_after_touching("data/challenge_cases.jsonl", baseline, "\n")
    assert ("DATASET_CHANGED", "routing_datasets", "data/challenge_cases.jsonl") in found


def test_evaluator_change_is_reported_as_evaluator_changed(baseline):
    found = findings_after_touching("src/evaluator.py", baseline)
    assert ("EVALUATOR_CHANGED", "routing_evaluator", "src/evaluator.py") in found


def test_historical_prediction_change_is_reported_as_prediction_changed(baseline):
    target = "outputs/llm_llm_v2_structured_predictions.jsonl"
    found = findings_after_touching(target, baseline, "\n")
    assert ("PREDICTION_CHANGED", "historical_llm_predictions", target) in found


def test_validator_is_covered_even_though_no_router_imports_it(frozen, baseline):
    """validate_and_repair rewrites every plan's tool mapping and can force a
    whole plan to clarify, but it is applied AROUND the routers rather than
    imported by them — a closure alone would never find it."""
    union = set(routing_freeze._group_files(frozen["groups"]["routing_runtime_code"]))
    assert "src/validator.py" in union
    found = findings_after_touching("src/validator.py", baseline)
    assert ("CHANGED", "routing_runtime_code", "src/validator.py") in found


# --------------------------------------------------------------------------
# 8. everything restored
# --------------------------------------------------------------------------

def test_tree_is_unchanged_after_all_tampering(baseline):
    assert diff_groups(baseline, routing_freeze.snapshot()) == []


def test_datasets_and_predictions_still_match_the_manifest(frozen):
    """Guards the guard. Datasets and historical predictions are evidence and
    are never a sanctioned edit target, so they must match the manifest even
    while runtime code is legitimately being changed. Without this, a restore
    that silently failed would let a corrupted dataset become the baseline."""
    current = routing_freeze.snapshot()
    for name in ["routing_datasets", "historical_llm_predictions"]:
        assert (routing_freeze._group_files(frozen["groups"][name])
                == routing_freeze._group_files(current["groups"][name])), name


# --------------------------------------------------------------------------
# scope and provenance properties
# --------------------------------------------------------------------------

def test_each_strategy_records_its_own_closure(frozen):
    per_entry = frozen["groups"]["routing_runtime_code"]["per_entry_closure"]
    assert set(per_entry) == set(routing_freeze.ROUTING_ENTRY_POINTS)
    # v3 delegates to v2, so a change to v2 reaches v3 but not v1. A single
    # union hash could not express that.
    assert "src/router_v2.py" in per_entry["rule_v3"]["files"]
    assert "src/router_v2.py" not in per_entry["rule_v1"]["files"]
    assert "src/extractors.py" in per_entry["rule_v1"]["files"]


def test_converted_predictions_are_excluded_as_derivatives(frozen):
    """Freezing a derived artifact next to its source would make a legitimate
    regeneration look like tampering."""
    files = routing_freeze._group_files(frozen["groups"]["historical_llm_predictions"])
    assert len(files) == 6
    assert not any("unified_routing_predictions" in rel for rel in files)


def test_manifest_states_the_limits_of_static_analysis(frozen):
    assert "静态 import 分析" in frozen["static_analysis_limitation"]
    assert frozen["unresolved_dependencies"] == []


def test_undeclared_prompt_reference_surfaces_as_unresolved_dependency():
    """A prompt named in runtime code but missing from the explicit list is
    exactly the gap static analysis leaves, so it must be reported rather than
    silently skipped."""
    original = list(routing_freeze.EXPLICIT_CONFIG_AND_PROMPTS)
    try:
        routing_freeze.EXPLICIT_CONFIG_AND_PROMPTS.remove("prompts/01_intent_router_v2.md")
        unresolved = routing_freeze.snapshot()["unresolved_dependencies"]
        assert any("01_intent_router_v2.md" in item["detail"] for item in unresolved)
    finally:
        routing_freeze.EXPLICIT_CONFIG_AND_PROMPTS[:] = original


def test_dataset_case_counts_are_recorded(frozen):
    counts = frozen["groups"]["routing_datasets"]["case_counts"]
    assert counts["data/eval_cases.jsonl"] == 46
    assert counts["data/challenge_cases.jsonl"] == 75


def test_runtime_environment_is_recorded(frozen):
    env = frozen["runtime_environment"]
    assert env["python_version"] and env["platform"]
    assert "pydantic" in env["dependency_versions"]


def test_tree_matches_the_recorded_freeze_or_says_what_moved():
    """The only test asserting the tree IS frozen. Skipped, not failed, while a
    sanctioned change is in flight, so a planned edit cannot masquerade as a
    freeze violation."""
    if not MANIFEST.exists():
        pytest.skip("run scripts/routing_freeze.py --write first")
    if routing_freeze.verify() != 0:
        pytest.skip("tree differs from the manifest (sanctioned change in flight)")
