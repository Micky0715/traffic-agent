from __future__ import annotations

import argparse
import json

from src.pipeline import run_pipeline
from src.router_v1 import route_v1
from src.router_v2 import route_v2
from src.router_v3 import route_v3
from src.validator import validate_and_repair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--router", choices=["v1", "v2", "v3"], default="v2")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.execute:
        output = run_pipeline(args.query)
    else:
        if args.router == "v1":
            plan = route_v1(args.query)
        elif args.router == "v2":
            plan = validate_and_repair(route_v2(args.query))
        else:
            plan = validate_and_repair(route_v3(args.query))
        output = plan.model_dump(mode="json")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
