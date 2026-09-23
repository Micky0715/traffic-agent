"""CLI for the structured-BM25 slice.

    python -m src.run_structured_rag_query --query "查询A16风机的功率" \
        --rag-mode structured_bm25

Thin by design: it parses arguments and prints. Every decision is made by
src.rag.service.StructuredRagService, the same object the tests exercise — a
demo path with its own copy of the logic would only prove that the demo works.

The mode flag is required to be `structured_bm25` explicitly, or the command
refuses. Default configuration is `legacy`, and nothing in the existing
pipeline reaches this code.
"""
from __future__ import annotations

import argparse
import json
import sys

from src.multimodal.config import load_multimodal_config
from src.multimodal.executor import LiveCallNotAuthorised, VisionReviewExecutor
from src.rag.config import load_rag_config
from src.rag.service import StructuredRagService


def main() -> None:
    parser = argparse.ArgumentParser(description="Structured BM25 RAG query")
    parser.add_argument("--query", required=True)
    parser.add_argument("--rag-mode", choices=["legacy", "structured_bm25"],
                        default=None,
                        help="overrides configs/rag.yaml rag.mode for this run")
    parser.add_argument("--task-type", default="drawing_field_query")
    parser.add_argument("--json", action="store_true", help="print the full result")
    parser.add_argument("--vision-review-mode",
                        choices=["disabled", "cache_replay", "live"], default=None,
                        help="overrides configs/multimodal.yaml multimodal_review.mode")
    parser.add_argument("--allow-live-vlm", action="store_true",
                        help="second switch required before any paid VLM call; "
                             "the config must ALSO set allow_live_vlm: true")
    args = parser.parse_args()

    cfg = load_rag_config()
    mode = args.rag_mode or cfg.mode
    if mode != "structured_bm25":
        print(f"rag mode is {mode!r}; this entry point only runs in "
              "'structured_bm25'. Pass --rag-mode structured_bm25 or set "
              "rag.mode in configs/rag.yaml.")
        sys.exit(2)

    # Vision review is constructed here so an unauthorised --vision-review-mode
    # live fails before the query runs, rather than half-way through it.
    mm = load_multimodal_config()
    review_mode = args.vision_review_mode or mm.review.mode
    if review_mode != mm.review.mode:
        mm = load_multimodal_config().__class__(
            review=type(mm.review)(**{**mm.review.__dict__, "mode": review_mode}),
            cache=mm.cache)
    try:
        executor = VisionReviewExecutor(mm, allow_live=args.allow_live_vlm)
    except LiveCallNotAuthorised as exc:
        print(f"refused: {exc}")
        sys.exit(3)

    service = StructuredRagService(cfg)
    service.load_corpus()
    result = service.answer(args.query, task_type=args.task_type)

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return

    decision = result.decision
    print(f"query      : {result.query}")
    print(f"plan       : entity={result.query_plan['asset_id']} "
          f"fields={result.query_plan['required_fields']} "
          f"unmatched={result.query_plan['unmatched_field_terms']}")
    print(f"entity     : {result.entity_resolution}")
    print(f"retrieved  : {result.retrieved_candidate_count}   "
          f"eligible: {result.eligible_evidence_count}")
    print(f"DECISION   : {decision['decision']}  "
          f"(routing={decision['routing_decision']})")
    print(f"reasons    : {decision['reason_codes']}")
    if decision["answerable_fields"]:
        print(f"answers    : {decision['answerable_fields']}")
    if decision["missing_fields"]:
        print(f"missing    : {decision['missing_fields']}")
    if decision["conflicts"]:
        print(f"conflicts  : {decision['conflicts']}")
    for ref in decision["evidence_refs"]:
        print(f"  evidence : {ref['document_id']} p{ref['page']} "
              f"bbox={ref['bbox']} chunk={ref['chunk_id']} src={ref.get('source')}")
    if result.rejected_candidates:
        print("rejected   :")
        for row in result.rejected_candidates[:5]:
            print(f"  [{row['bm25_rank']}] score={row['bm25_score']:.4f} "
                  f"{row['text'][:52]!r} -> {row['rejected_because']}")
    print(f"latency    : {result.latency_ms:.2f} ms")
    print(f"vision review: mode={review_mode} "
          f"cache_entries={len(executor.cache)} "
          f"max_live_calls={mm.review.max_live_calls}")
    if review_mode == "disabled":
        print("             no vision model was called (mode=disabled)")


if __name__ == "__main__":
    main()
