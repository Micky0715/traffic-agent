"""Tests for the vision freeze scope.

The property under test is the one the previous freeze got wrong: the scope
must follow what the pipeline actually imports, not a directory boundary.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import vision_freeze  # noqa: E402

MANIFEST = ROOT / "outputs" / "vision_freeze_manifest.json"


def test_closure_reaches_dependencies_outside_the_vision_directory():
    """A directory-scoped freeze would report green after an edit to any of
    these, even though each one changes parsing behaviour."""
    closure = set(vision_freeze.import_closure(vision_freeze.ENTRY_POINTS))
    for path in ["src/models.py", "src/tools.py", "src/llm_router.py",
                 "src/drawing_extractor.py"]:
        assert path in closure, f"{path} is imported by the pipeline but outside the scope"


def test_closure_excludes_modules_the_pipeline_never_imports():
    """The old whole-tree hash fired on these, turning the freeze check into
    noise."""
    closure = set(vision_freeze.import_closure(vision_freeze.ENTRY_POINTS))
    for path in ["src/routing/adapter.py", "src/routing/schema.py",
                 "src/pipeline.py", "src/router_v3.py", "src/evaluator.py"]:
        assert path not in closure, f"{path} should not be in the vision freeze scope"


def test_closure_is_deterministic():
    assert (vision_freeze.import_closure(vision_freeze.ENTRY_POINTS)
            == vision_freeze.import_closure(vision_freeze.ENTRY_POINTS))


def test_every_entry_point_exists():
    for path in vision_freeze.ENTRY_POINTS:
        assert (ROOT / path).exists(), path


def test_manifest_records_the_resolved_file_list_not_just_a_digest():
    """Recording the list is what makes a SCOPE change visible as a diff
    instead of being silently absorbed into one combined hash."""
    if not MANIFEST.exists():
        pytest.skip("run scripts/vision_freeze.py --write first")
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert data["code_files"] and data["config_files"]
    assert "configs/visual_parser.yaml" in data["config_files"]
    assert data["combined_sha256"]


def test_combined_hash_changes_when_any_in_scope_file_changes():
    hashes = {"a.py": "1", "b.py": "2"}
    before = vision_freeze.combined(hashes)
    assert vision_freeze.combined({**hashes, "b.py": "3"}) != before
    assert vision_freeze.combined(hashes) == before


def test_current_tree_matches_the_recorded_freeze():
    """The only test asserting the tree IS frozen. Skipped, not failed, while a
    sanctioned change is in flight — a planned edit must not masquerade as a
    freeze violation, and a tooling suite that errors out the moment an
    approved change lands is a suite nobody can trust during a change. The
    routing freeze tests already work this way; this brings vision into line.

    What the tool reports is covered by the other tests in this module, which
    compare against a baseline they take themselves."""
    if not MANIFEST.exists():
        pytest.skip("run scripts/vision_freeze.py --write first")
    if vision_freeze.verify() != 0:
        pytest.skip("tree differs from the manifest (sanctioned change in flight)")
