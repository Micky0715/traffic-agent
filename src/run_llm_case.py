from __future__ import annotations

import argparse
import json

from src.llm_router import route_with_llm
from src.validator import validate_and_repair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    args = parser.parse_args()
    plan = validate_and_repair(route_with_llm(args.query))
    print(json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
