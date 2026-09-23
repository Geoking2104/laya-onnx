"""Run the labeled decision eval and print a Markdown/JSON summary.

Usage:
    python benchmarks/evaluate.py <model_dir_or_hub_id> [--json benchmarks/results.json]

``<model>`` is a local ONNX bundle or an already-cached Hub id. Runs on CPU by
default; pass ``--deterministic`` for the reproducible path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from laya_onnx import load
from laya_onnx.eval import format_markdown, load_items, run_eval

HERE = Path(__file__).resolve().parent
DEFAULT_EVAL = HERE / "eval" / "decisions.jsonl"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="benchmarks/evaluate.py", description=__doc__)
    parser.add_argument("model", help="Local ONNX bundle or an already-cached Hub id")
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL, help="JSONL eval set")
    parser.add_argument("--providers", default="cpu")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--json", type=Path, help="Write the raw results JSON here")
    args = parser.parse_args(argv)

    items = load_items(args.eval)
    agent = load(
        args.model,
        providers=args.providers,
        threads=args.threads,
        deterministic=args.deterministic,
    )
    results = run_eval(agent, items)
    results["model"] = {
        "id": str(args.model),
        "providers": args.providers,
        "threads": args.threads,
        "deterministic": bool(args.deterministic),
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    print(format_markdown(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
