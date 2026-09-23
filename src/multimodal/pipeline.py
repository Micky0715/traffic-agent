"""The closed loop, as one reusable object.

    StructuredRagService.answer()            <- first decision
      -> targets for what it could not answer
      -> VisionReviewExecutor                <- crop, then disabled/replay/live
      -> evidence_from_review + merge_evidence
      -> RequiredEvidencePolicy              <- second decision

This lives here, not in the CLI, so the command line stays a printer. A demo
that carries its own copy of the orchestration only ever proves the demo works.

The loop runs at most once. `remediation_exhausted` is set before the second
decision so nothing downstream can ask for another review round: an agent that
keeps re-cropping a page it cannot read spends money and still cannot read it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.multimodal.config import MultimodalConfig
from src.multimodal.crop import sha256_file
from src.multimodal.decision import run_second_pass
from src.multimodal.executor import VisionReviewExecutor
from src.multimodal.fusion import evidence_from_review, fuse
from src.multimodal.reasons import INVALID_FIELD_VALUE, REQUIRED_FIELD_MISSING
from src.multimodal.schemas import CropResult, ReviewOutcome, ReviewTarget, VisionReviewResult
from src.multimodal.triggers import TriggerCollector
from src.rag.policy import Decision, EvidenceBundle, RequiredEvidencePolicy
from src.vision.schemas import FieldEvidence

ROOT = Path(__file__).resolve().parents[2]


def locate_label_bbox(field_name: str, ocr_blocks: Sequence) -> List[float]:
    """Where the label for `field_name` sits, if OCR read it at all.

    Matching is on the label text, never on the field's expected VALUE — using
    the value would mean the crop is placed by knowing the answer, and a
    region chosen that way can only confirm what was already believed.
    """
    for block in ocr_blocks:
        text = getattr(block, "text", "") or ""
        if field_name and field_name in text:
            bbox = list(getattr(block, "bbox", []) or [])
            if len(bbox) == 4:
                return [float(v) for v in bbox]
    return []


class MultimodalReviewPipeline:
    def __init__(
        self,
        config: MultimodalConfig,
        policy: RequiredEvidencePolicy,
        *,
        allow_live: bool = False,
        client_factory=None,
    ):
        self.config = config
        self.cfg = config.review
        self.policy = policy
        self.executor = VisionReviewExecutor(
            config, allow_live=allow_live, client_factory=client_factory)

    # -- what deserves a look ---------------------------------------------

    def targets_for(
        self,
        decision: Decision,
        *,
        document_id: str,
        drawing_type: Optional[str] = None,
        page: int = 1,
        ocr_blocks: Sequence = (),
        invalid_fields: Sequence[str] = (),
        evidence_refs: Sequence[dict] = (),
    ) -> List[ReviewTarget]:
        """Only what the first decision could not settle."""
        collector = TriggerCollector(document_id, drawing_type=drawing_type, page=page)
        ref_by_field = {r.get("field_name"): r for r in evidence_refs
                        if r.get("field_name")}

        for name in decision.missing_fields:
            collector.add(name, REQUIRED_FIELD_MISSING,
                          label_bbox=locate_label_bbox(name, ocr_blocks))
        for name in invalid_fields:
            ref = ref_by_field.get(name, {})
            collector.add(name, INVALID_FIELD_VALUE,
                          ocr_value=ref.get("value"),
                          value_bbox=ref.get("bbox"),
                          chunk_id=ref.get("chunk_id"))
        return collector.targets(self.cfg)[:self.cfg.max_live_calls] \
            if self.cfg.mode == "live" else collector.targets(self.cfg)

    # -- run it ------------------------------------------------------------

    def review_all(
        self, targets: Sequence[ReviewTarget], image_path: Path,
    ) -> List[Tuple[ReviewTarget, VisionReviewResult, Optional[CropResult]]]:
        out = []
        for target in targets:
            crop = self.executor.crop_for(target, image_path)
            result = self.executor.review_with_crop(target, crop)
            out.append((target, result, crop if crop.available else None))
        return out

    def run(
        self,
        *,
        document_id: str,
        image_path: Path,
        bundle_before: EvidenceBundle,
        decision_before: Decision,
        ocr_evidence: Sequence[FieldEvidence],
        value_validation_cfg,
        drawing_type: Optional[str] = None,
        drawing_type_spec=None,
        page: int = 1,
        ocr_blocks: Sequence = (),
        invalid_fields: Sequence[str] = (),
        evidence_refs: Sequence[dict] = (),
    ) -> ReviewOutcome:
        targets = self.targets_for(
            decision_before, document_id=document_id, drawing_type=drawing_type,
            page=page, ocr_blocks=ocr_blocks, invalid_fields=invalid_fields,
            evidence_refs=evidence_refs)

        if not targets:
            # Nothing was reviewed, so there is nothing new to decide on.
            # Running the second pass anyway sets remediation_exhausted and
            # measurably downgraded a clean page from execute to partial —
            # punishing it for a remediation it never needed.
            return ReviewOutcome(
                document_id=document_id,
                decision_before_review=decision_before.decision,
                decision_after_review=decision_before.decision,
                notes={"targets": 0, "review_rounds": 0,
                       "second_pass_skipped": "no review was triggered",
                       "inference_mode": self.cfg.mode})

        reviewed = self.review_all(targets, image_path)
        vlm_evidence = evidence_from_review(
            reviewed, value_validation_cfg, drawing_type_spec)
        fused = fuse(ocr_evidence, vlm_evidence)

        outcome = run_second_pass(
            policy=self.policy, bundle_before=bundle_before,
            decision_before=decision_before, fused=fused,
            review_results=[r for _, r, _ in reviewed], cfg=self.cfg,
            document_id=document_id, page=page)
        outcome.notes.update({
            "targets": len(targets),
            "image_sha256": sha256_file(image_path) if image_path.exists() else None,
            "inference_mode": self.cfg.mode,
            "live_calls_made": self.executor.live_calls_made,
            "crops": [
                {"field_name": t.field_name, "region_scope": c.region_scope,
                 "crop_path": c.crop_path, "crop_sha256": c.crop_sha256,
                 "bbox_original": c.bbox_original,
                 "bbox_with_padding": c.bbox_with_padding,
                 "page": c.page, "review_reasons": c.review_reasons}
                for t, _, c in reviewed if c is not None],
        })
        return outcome
