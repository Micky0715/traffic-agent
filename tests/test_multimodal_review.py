"""Crop -> vision review -> fusion -> re-decision.

Every test that touches a model uses a stub client. NOTHING here calls a paid
API; the live path is exercised only through a fake whose replies are written
in this file. Tests marked REAL IMAGE cut from an actual drawing in
data/drawings and are real in that sense only.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.multimodal.cache import (
    CacheVerificationError, VisionReviewCache, make_cache_key,
)
from src.multimodal.config import (
    CacheConfig, MultimodalConfig, MultimodalReviewConfig, load_multimodal_config,
)
from src.multimodal.crop import CropSpaceError, resolve_review_crop, sha256_file
from src.multimodal.decision import apply_single_source_guard, to_policy_fields
from src.multimodal.executor import LiveCallNotAuthorised, VisionReviewExecutor, summarize
from src.multimodal.fusion import evidence_from_review, fuse, single_source_fields
from src.multimodal.reasons import (
    ALL_REASONS, CELL_ASSIGNMENT_UNCERTAIN, INVALID_FIELD_VALUE,
    LOW_OCR_CONFIDENCE_ON_REQUIRED_FIELD, REQUIRED_FIELD_MISSING, from_legacy,
)
from src.multimodal.response_schema import SchemaValidationError, validate_response
from src.multimodal.schemas import CropResult, ReviewTarget, VisionReviewResult
from src.multimodal.triggers import collect_targets
from src.vision.schemas import FieldEvidence, FieldPair

ROOT = Path(__file__).resolve().parents[1]
DRAWING = ROOT / "data" / "drawings" / "FAN-MULTI-01.png"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def make_config(tmp_path, **overrides) -> MultimodalConfig:
    base = load_multimodal_config()
    review = MultimodalReviewConfig(**{
        **base.review.__dict__, **overrides})
    return MultimodalConfig(
        review=review,
        cache=CacheConfig(path=tmp_path / "cache.json",
                          crop_dir=tmp_path / "crops"))


def target(**kwargs) -> ReviewTarget:
    defaults = dict(
        request_id="review:DOC:控制柜编号", document_id="DOC",
        field_name="控制柜编号", canonical_field_name="控制柜编号",
        review_reasons=[REQUIRED_FIELD_MISSING])
    return ReviewTarget(**{**defaults, **kwargs})


class StubClient:
    """Stands in for the OpenAI-compatible gateway. Never touches a network."""

    def __init__(self, payload, *, raise_exc=None):
        self._payload = payload
        self._raise = raise_exc
        self.calls = 0
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls += 1
        if self._raise:
            raise self._raise
        content = self._payload if isinstance(self._payload, str) \
            else json.dumps(self._payload, ensure_ascii=False)

        class _Msg:
            def __init__(self, c): self.content = c

        class _Choice:
            def __init__(self, c): self.message = _Msg(c)

        class _Resp:
            def __init__(self, c): self.choices = [_Choice(c)]

        return _Resp(content)


def good_payload(value="FAN-CAB-23", field="控制柜编号"):
    return {"request_id": "review:DOC:控制柜编号",
            "fields": [{"canonical_field_name": field, "raw_name": field,
                        "raw_value": value, "readable": True,
                        "evidence_bbox_in_crop": [10, 8, 120, 40],
                        "notes": None}],
            "unreadable_reasons": []}


def unreadable_payload(field="控制柜编号"):
    return {"request_id": "review:DOC:控制柜编号",
            "fields": [{"canonical_field_name": field, "raw_name": None,
                        "raw_value": None, "readable": False,
                        "evidence_bbox_in_crop": [], "notes": "watermark_occlusion"}],
            "unreadable_reasons": ["watermark_occlusion"]}


# --------------------------------------------------------------------------
# 裁剪策略
# --------------------------------------------------------------------------

def test_value_bbox_is_cropped_when_known(tmp_path):
    """REAL IMAGE. A known value box wins over everything else."""
    cfg = make_config(tmp_path)
    crop = resolve_review_crop(
        target(value_bbox=[200, 300, 420, 340], cell_bbox=[0, 0, 900, 700]),
        DRAWING, cfg.review, tmp_path / "crops")
    assert crop.available
    assert crop.region_scope == "value"
    assert crop.bbox_original == [200.0, 300.0, 420.0, 340.0]
    assert Path(crop.crop_path).exists()


def test_a_lone_label_is_extended_to_its_right(tmp_path):
    """REAL IMAGE. The value is missing; it normally sits to the right of its
    label, so that is where the crop looks — and the label stays in frame."""
    cfg = make_config(tmp_path)
    crop = resolve_review_crop(
        target(label_bbox=[100, 200, 180, 230]),
        DRAWING, cfg.review, tmp_path / "crops")
    assert crop.available
    assert crop.region_scope == "label_right"
    # Extends right, and the label's own left edge is still inside the crop.
    assert crop.bbox_with_padding[2] > 180 + cfg.review.crop_padding_px
    assert crop.bbox_with_padding[0] <= 100


def test_row_crop_when_only_the_row_is_known(tmp_path):
    """REAL IMAGE. Coarser, and region_scope says so."""
    cfg = make_config(tmp_path)
    crop = resolve_review_crop(
        target(row_bbox=[60, 400, 800, 440]),
        DRAWING, cfg.review, tmp_path / "crops")
    assert crop.available
    assert crop.region_scope == "row"


def test_out_of_bounds_bbox_is_clipped_not_rejected(tmp_path):
    """REAL IMAGE. A box running off the edge is a rounding problem; clip it."""
    cfg = make_config(tmp_path)
    crop = resolve_review_crop(
        target(value_bbox=[-50, -40, 300, 200]),
        DRAWING, cfg.review, tmp_path / "crops")
    assert crop.available
    assert crop.bbox_with_padding[0] == 0.0 and crop.bbox_with_padding[1] == 0.0


def test_a_bbox_from_another_coordinate_space_is_refused(tmp_path):
    """REAL IMAGE. A box whose ORIGIN is past the image cannot be a rounding
    error — it was measured on a different (preprocessed) picture. Clipping it
    would produce a plausible-looking crop of the wrong place."""
    cfg = make_config(tmp_path)
    crop = resolve_review_crop(
        target(value_bbox=[99999, 99999, 100200, 100040]),
        DRAWING, cfg.review, tmp_path / "crops")
    assert not crop.available
    assert crop.region_scope == "unavailable"
    assert "coordinate space" in crop.reason


def test_no_bbox_at_all_never_falls_back_to_the_whole_page(tmp_path):
    cfg = make_config(tmp_path)
    crop = resolve_review_crop(target(), DRAWING, cfg.review, tmp_path / "crops")
    assert not crop.available
    assert crop.region_scope == "unavailable"


def test_a_degenerate_bbox_is_refused_not_stretched(tmp_path):
    """A zero-height box padded out is not a readable region."""
    cfg = make_config(tmp_path, crop_padding_px=0,
                      min_crop_width=64, min_crop_height=32)
    crop = resolve_review_crop(
        target(value_bbox=[100, 100, 100, 100]),
        DRAWING, cfg.review, tmp_path / "crops")
    assert not crop.available


def test_crop_hash_is_stable_across_runs(tmp_path):
    """REAL IMAGE. The same region cut twice is byte-identical, which is what
    makes the cache key meaningful at all."""
    cfg = make_config(tmp_path)
    box = [200, 300, 420, 340]
    a = resolve_review_crop(target(value_bbox=box), DRAWING, cfg.review, tmp_path / "a")
    b = resolve_review_crop(target(value_bbox=box), DRAWING, cfg.review, tmp_path / "b")
    assert a.crop_sha256 == b.crop_sha256
    assert a.source_image_sha256 == sha256_file(DRAWING)


# --------------------------------------------------------------------------
# 缓存校验
# --------------------------------------------------------------------------

def _seed_cache(tmp_path, **overrides):
    cache = VisionReviewCache(tmp_path / "cache.json")
    key = "k1"
    result = VisionReviewResult(request_id="r", status="success",
                                inference_mode="live")
    cache.set(key, result, input_image_sha256="IMG", crop_sha256="CROP",
              model_name="qwen-vl-max", prompt_version="vision_field_review_v1",
              schema_version="vision_review_v1", crop_path="c.png",
              document_id="DOC", field_names=["控制柜编号"])
    return cache, key


def test_a_matching_cache_entry_replays_and_is_labelled_replay(tmp_path):
    cache, key = _seed_cache(tmp_path)
    got = cache.get(key, expected_image_sha256="IMG", expected_crop_sha256="CROP",
                    expected_model="qwen-vl-max",
                    expected_prompt_version="vision_field_review_v1",
                    expected_schema_version="vision_review_v1")
    assert got is not None
    # The recorded run was live. Replaying it is not, and cannot be reported so.
    assert got.inference_mode == "cache_replay"


@pytest.mark.parametrize("field,value", [
    ("expected_crop_sha256", "DIFFERENT"),
    ("expected_image_sha256", "DIFFERENT"),
    ("expected_model", "some-other-model"),
    ("expected_prompt_version", "vision_field_review_v2"),
    ("expected_schema_version", "vision_review_v2"),
])
def test_any_provenance_mismatch_raises_rather_than_missing(tmp_path, field, value):
    """A silent miss would be retried live or reported as "no cached evidence".
    Both hide that the recording no longer describes these inputs."""
    cache, key = _seed_cache(tmp_path)
    kwargs = dict(expected_image_sha256="IMG", expected_crop_sha256="CROP",
                  expected_model="qwen-vl-max",
                  expected_prompt_version="vision_field_review_v1",
                  expected_schema_version="vision_review_v1")
    kwargs[field] = value
    with pytest.raises(CacheVerificationError):
        cache.get(key, **kwargs)


def test_an_absent_entry_is_a_miss_not_an_error(tmp_path):
    cache, _ = _seed_cache(tmp_path)
    assert cache.get("nope", expected_image_sha256="IMG",
                     expected_crop_sha256="CROP", expected_model="qwen-vl-max",
                     expected_prompt_version="vision_field_review_v1",
                     expected_schema_version="vision_review_v1") is None


def test_cache_entry_records_its_own_provenance(tmp_path):
    """The defect this cache was written to fix: the old one keyed on a hash of
    the provenance and stored none of it, so an entry could not be audited."""
    cache, key = _seed_cache(tmp_path)
    entry = json.loads((tmp_path / "cache.json").read_text(encoding="utf-8"))[key]
    for name in ("input_image_sha256", "crop_sha256", "model_name",
                 "prompt_version", "schema_version", "recorded_at",
                 "recorded_from"):
        assert entry[name], f"{name} missing from cache entry"
    assert cache.provenance_summary()["recorded_from_live_inference"] == 1


# --------------------------------------------------------------------------
# 执行器与授权
# --------------------------------------------------------------------------

def test_disabled_mode_does_not_invoke(tmp_path):
    cfg = make_config(tmp_path, mode="disabled")
    client = StubClient(good_payload())
    executor = VisionReviewExecutor(cfg, client_factory=lambda: client)
    result = executor.review(target(value_bbox=[200, 300, 420, 340]), DRAWING)
    assert result.status == "not_invoked"
    assert result.inference_mode == "disabled"
    assert client.calls == 0


def test_live_mode_without_authorisation_is_refused(tmp_path):
    """Config alone is not enough; the flag alone is not enough."""
    cfg = make_config(tmp_path, mode="live", allow_live_vlm=True)
    with pytest.raises(LiveCallNotAuthorised):
        VisionReviewExecutor(cfg, allow_live=False)
    cfg2 = make_config(tmp_path, mode="live", allow_live_vlm=False)
    with pytest.raises(LiveCallNotAuthorised):
        VisionReviewExecutor(cfg2, allow_live=True)


def test_cache_replay_miss_is_not_upgraded_to_a_call(tmp_path):
    cfg = make_config(tmp_path, mode="cache_replay")
    client = StubClient(good_payload())
    executor = VisionReviewExecutor(cfg, client_factory=lambda: client)
    result = executor.review(target(value_bbox=[200, 300, 420, 340]), DRAWING)
    assert result.status == "not_invoked"
    assert result.error_type == "no_cache_entry"
    assert client.calls == 0


def test_max_live_calls_stops_the_run(tmp_path):
    cfg = make_config(tmp_path, mode="live", allow_live_vlm=True, max_live_calls=2)
    client = StubClient(good_payload())
    executor = VisionReviewExecutor(cfg, allow_live=True, client_factory=lambda: client)
    statuses = []
    for i in range(4):
        # Distinct regions so each one misses the cache and wants a call.
        t = target(request_id=f"r{i}", value_bbox=[200 + i * 40, 300, 420 + i * 40, 340])
        statuses.append(executor.review(t, DRAWING).status)
    assert client.calls == 2
    assert statuses[2:] == ["budget_exhausted", "budget_exhausted"]


def test_crop_unavailable_short_circuits_before_any_call(tmp_path):
    cfg = make_config(tmp_path, mode="live", allow_live_vlm=True)
    client = StubClient(good_payload())
    executor = VisionReviewExecutor(cfg, allow_live=True, client_factory=lambda: client)
    result = executor.review(target(), DRAWING)
    assert result.status == "crop_unavailable"
    assert client.calls == 0


def test_a_live_call_is_recorded_then_replayable(tmp_path):
    cfg_live = make_config(tmp_path, mode="live", allow_live_vlm=True)
    client = StubClient(good_payload())
    live = VisionReviewExecutor(cfg_live, allow_live=True, client_factory=lambda: client)
    first = live.review(target(value_bbox=[200, 300, 420, 340]), DRAWING)
    assert first.status == "success" and first.inference_mode == "live"

    cfg_replay = make_config(tmp_path, mode="cache_replay")
    replay = VisionReviewExecutor(cfg_replay, client_factory=lambda: client)
    second = replay.review(target(value_bbox=[200, 300, 420, 340]), DRAWING)
    assert second.status == "success"
    assert second.inference_mode == "cache_replay"
    assert client.calls == 1


# --------------------------------------------------------------------------
# 输出 Schema
# --------------------------------------------------------------------------

def test_schema_failure_is_recorded_not_retried_forever(tmp_path):
    cfg = make_config(tmp_path, mode="live", allow_live_vlm=True)
    client = StubClient("这张图我看不太清，可能是 FAN-CAB-23 吧")
    executor = VisionReviewExecutor(cfg, allow_live=True, client_factory=lambda: client)
    result = executor.review(target(value_bbox=[200, 300, 420, 340]), DRAWING)
    assert result.status == "schema_failure"
    assert client.calls == 1


def test_a_reply_about_an_unrequested_field_is_rejected():
    with pytest.raises(SchemaValidationError):
        validate_response(good_payload(field="电机编号"), ["控制柜编号"])


def test_readable_true_with_no_value_is_rejected():
    payload = {"fields": [{"canonical_field_name": "控制柜编号", "raw_value": "",
                           "readable": True, "evidence_bbox_in_crop": []}]}
    with pytest.raises(SchemaValidationError):
        validate_response(payload, ["控制柜编号"])


def test_unreadable_reply_is_a_valid_answer(tmp_path):
    cfg = make_config(tmp_path, mode="live", allow_live_vlm=True)
    client = StubClient(unreadable_payload())
    executor = VisionReviewExecutor(cfg, allow_live=True, client_factory=lambda: client)
    result = executor.review(target(value_bbox=[200, 300, 420, 340]), DRAWING)
    assert result.status == "unreadable"
    assert result.fields[0].readable is False
    assert result.unreadable_reasons == ["watermark_occlusion"]


# --------------------------------------------------------------------------
# 触发
# --------------------------------------------------------------------------

def test_a_complete_valid_field_triggers_nothing():
    """The clean page. No reason fires, so no call is bought."""
    cfg = load_multimodal_config().review
    targets = collect_targets(
        document_id="CLEAN", drawing_type="wiring_diagram", cfg=cfg,
        pairs=[FieldPair(field_name="控制柜编号", raw_value="FAN-CAB-23",
                         confidence=0.95, bbox=[1, 2, 3, 4])],
        required_fields=["控制柜编号"])
    assert targets == []


def test_one_field_with_two_reasons_is_one_target():
    cfg = load_multimodal_config().review
    targets = collect_targets(
        document_id="D", drawing_type=None, cfg=cfg,
        pairs=[FieldPair(field_name="控制柜编号", raw_value="FAN-CAB-2?",
                         confidence=0.3, bbox=[1, 2, 3, 4])],
        required_fields=["控制柜编号"],
        invalid_field_labels=["控制柜编号"])
    assert len(targets) == 1
    assert set(targets[0].review_reasons) == {
        INVALID_FIELD_VALUE, LOW_OCR_CONFIDENCE_ON_REQUIRED_FIELD}


def test_a_missing_required_field_outranks_an_ambiguous_cell():
    """max_live_calls truncates this list, so order decides what the budget
    buys. In this repo ambiguous cells outnumber everything 57 to 3."""
    cfg = load_multimodal_config().review
    targets = collect_targets(
        document_id="D", drawing_type=None, cfg=cfg,
        required_fields=["控制柜编号"],
        uncertain_cells=[{"field_name": "备注", "cell_bbox": [0, 0, 10, 10]}])
    assert [t.review_reasons[0] for t in targets] == [
        REQUIRED_FIELD_MISSING, CELL_ASSIGNMENT_UNCERTAIN]


def test_every_legacy_table_reason_maps_onto_this_vocabulary():
    from src.tables.review import REVIEW_REASONS
    for legacy in REVIEW_REASONS:
        assert from_legacy(legacy) in ALL_REASONS


def test_an_unmapped_legacy_reason_raises():
    with pytest.raises(ValueError):
        from_legacy("SOMETHING_NEW")


# --------------------------------------------------------------------------
# 合流
# --------------------------------------------------------------------------

def ev(source, field, raw, normalized=None, status="valid", entity_id=None):
    return FieldEvidence(
        entity_id=entity_id, occurrence_id=f"{source}:{field}:{raw}",
        field_name=field, raw_value=raw,
        normalized_value=normalized if normalized is not None else raw,
        source=source, validation_status=status)


def test_identical_values_are_not_a_conflict_and_both_survive():
    merged = fuse([ev("ocr", "电机编号", "M-13")], [ev("vlm", "电机编号", "M-13")])
    assert merged.conflicts == []
    assert len(merged.evidence) == 2


def test_a_formatting_only_difference_is_not_a_conflict():
    merged = fuse([ev("ocr", "电机编号", "M - 13", "M-13")],
                  [ev("vlm", "电机编号", "M-13", "M-13")])
    assert merged.conflicts == []
    raws = {e.raw_value for e in merged.evidence}
    assert raws == {"M - 13", "M-13"}       # both raw forms kept verbatim


def test_different_values_conflict_and_are_left_unresolved():
    merged = fuse([ev("ocr", "图号", "FAN-F53-01")],
                  [ev("vlm", "图号", "FAN-FS3-01")])
    assert len(merged.conflicts) == 1
    conflict = merged.conflicts[0]
    assert conflict.ocr_value == "FAN-F53-01"
    assert conflict.vlm_value == "FAN-FS3-01"
    # Both sides stay on the record; nothing picked a winner.
    assert len(merged.evidence) == 2


def test_ocr_missing_vlm_present_is_single_source_not_a_conflict():
    merged = fuse([], [ev("vlm", "断路器编号", "QF-23")])
    assert merged.conflicts == []
    assert single_source_fields(merged)["vlm_only"] == ["断路器编号"]


def test_a_vlm_value_goes_through_the_same_validation_as_ocr(tmp_path):
    """Not skipped because the model produced it."""
    from src.vision.config import load_config
    cfg = load_config().value_validation
    result = VisionReviewResult(
        request_id="r", status="success", inference_mode="cache_replay",
        fields=[{"canonical_field_name": "图号", "raw_value": "!!!bad!!!",
                 "readable": True, "evidence_bbox_in_crop": []}])
    evidence = evidence_from_review([(target(field_name="图号"), result, None)], cfg)
    assert len(evidence) == 1
    assert evidence[0].source == "vlm"
    assert evidence[0].validation_status in ("invalid", "unknown")
    # A model's self-reported certainty is not a business confidence.
    assert evidence[0].confidence == 0.0


def test_an_unreadable_field_produces_no_evidence():
    from src.vision.config import load_config
    cfg = load_config().value_validation
    result = VisionReviewResult(request_id="r", status="unreadable",
                                inference_mode="cache_replay")
    assert evidence_from_review([(target(), result, None)], cfg) == []


def test_a_crop_relative_bbox_is_mapped_back_to_page_coordinates():
    from src.vision.config import load_config
    cfg = load_config().value_validation
    crop = CropResult(available=True, region_scope="value",
                      bbox_original=[200, 300, 420, 340],
                      bbox_with_padding=[176, 276, 444, 364])
    result = VisionReviewResult(
        request_id="r", status="success", inference_mode="cache_replay",
        fields=[{"canonical_field_name": "控制柜编号", "raw_value": "FAN-CAB-23",
                 "readable": True, "evidence_bbox_in_crop": [10, 8, 120, 40]}])
    evidence = evidence_from_review([(target(), result, crop)], cfg)
    assert evidence[0].bbox == [186.0, 284.0, 296.0, 316.0]


# --------------------------------------------------------------------------
# 二次决策
# --------------------------------------------------------------------------

def test_a_conflict_is_handed_to_the_policy_as_a_conflict():
    merged = fuse([ev("ocr", "图号", "FAN-F53-01")],
                  [ev("vlm", "图号", "FAN-FS3-01")])
    fields = to_policy_fields(merged, document_id="D", page=1)
    assert len(fields) == 1
    assert fields[0].in_conflict
    assert fields[0].conflict_with == "FAN-FS3-01"


def test_a_high_risk_field_on_one_vlm_reading_cannot_execute():
    from src.rag.policy import Decision
    cfg = load_multimodal_config().review
    merged = fuse([], [ev("vlm", "控制柜编号", "FAN-CAB-23")])
    decision = Decision(decision="execute", routing_decision="execute")
    guarded, flagged = apply_single_source_guard(decision, merged, cfg)
    assert flagged == ["控制柜编号"]
    assert guarded.decision == cfg.single_source_vlm_high_risk_decision
    assert guarded.decision != "execute"
    assert "single_source_vlm_high_risk_field" in guarded.reason_codes


def test_a_corroborated_high_risk_field_is_not_downgraded():
    from src.rag.policy import Decision
    cfg = load_multimodal_config().review
    merged = fuse([ev("ocr", "控制柜编号", "FAN-CAB-23")],
                  [ev("vlm", "控制柜编号", "FAN-CAB-23")])
    decision = Decision(decision="execute", routing_decision="execute")
    guarded, flagged = apply_single_source_guard(decision, merged, cfg)
    assert flagged == []
    assert guarded.decision == "execute"


def test_summarize_separates_replay_from_inference():
    results = [
        VisionReviewResult(request_id="a", status="success", inference_mode="live"),
        VisionReviewResult(request_id="b", status="success", inference_mode="cache_replay"),
        VisionReviewResult(request_id="c", status="not_invoked", inference_mode="disabled"),
        VisionReviewResult(request_id="d", status="crop_unavailable", inference_mode="live"),
    ]
    stats = summarize(results)
    assert stats["real_inference"] == 1      # the crop_unavailable one is not a call
    assert stats["cache_replay"] == 1
    assert stats["disabled"] == 1
    assert stats["crop_unavailable"] == 1


def test_defaults_are_off():
    """A committed config must not be able to spend money on its own."""
    cfg = load_multimodal_config().review
    assert cfg.mode == "disabled"
    assert cfg.allow_live_vlm is False


def test_a_request_that_never_reached_the_cache_is_not_counted_as_a_replay():
    """Regression: 9 crop_unavailable requests in cache_replay mode were
    reported as `cache_replay: 11` when exactly 2 entries were read."""
    results = [
        VisionReviewResult(request_id="a", status="success",
                           inference_mode="cache_replay"),
        VisionReviewResult(request_id="b", status="crop_unavailable",
                           inference_mode="cache_replay"),
        VisionReviewResult(request_id="c", status="not_invoked",
                           inference_mode="cache_replay",
                           error_type="no_cache_entry"),
    ]
    stats = summarize(results)
    assert stats["cache_replay"] == 1
    assert stats["crop_unavailable"] == 1
    assert stats["total_review_requests"] == 3


def test_a_page_that_needed_no_review_is_not_downgraded(tmp_path):
    """Regression: the clean page went execute -> partial because the second
    pass ran anyway and set remediation_exhausted, punishing it for a
    remediation it never needed."""
    from src.multimodal.pipeline import MultimodalReviewPipeline
    from src.rag.config import load_rag_config
    from src.rag.policy import Decision, EvidenceBundle, RequiredEvidencePolicy

    cfg = make_config(tmp_path, mode="disabled")
    policy = RequiredEvidencePolicy(load_rag_config().evidence_policy)
    pipeline = MultimodalReviewPipeline(cfg, policy)

    decision = Decision(decision="execute", routing_decision="execute",
                        missing_fields=[])
    outcome = pipeline.run(
        document_id="CLEAN", image_path=DRAWING,
        bundle_before=EvidenceBundle(task_type="drawing_field_query",
                                     entity_id="CLEAN"),
        decision_before=decision, ocr_evidence=[],
        value_validation_cfg=None)
    assert outcome.decision_after_review == "execute"
    assert outcome.notes["review_rounds"] == 0
    assert outcome.notes["second_pass_skipped"]


def test_only_one_review_round_is_ever_run(tmp_path):
    """The bounded loop. An agent that keeps re-cropping a page it cannot read
    spends money and still cannot read it."""
    from src.multimodal.decision import run_second_pass
    from src.rag.config import load_rag_config
    from src.rag.policy import Decision, EvidenceBundle, RequiredEvidencePolicy

    cfg = load_multimodal_config().review
    policy = RequiredEvidencePolicy(load_rag_config().evidence_policy)
    bundle = EvidenceBundle(task_type="drawing_field_query", entity_id="D")
    outcome = run_second_pass(
        policy=policy, bundle_before=bundle,
        decision_before=Decision(decision="fallback"),
        fused=fuse([], [ev("vlm", "图号", "FAN-A23-01")]),
        review_results=[], cfg=cfg, document_id="D")
    assert outcome.notes["review_rounds"] == 1
    assert outcome.notes["review_rounds_capped_at"] == 1
