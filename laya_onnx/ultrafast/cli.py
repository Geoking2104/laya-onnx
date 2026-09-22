from __future__ import annotations
import argparse, json
from .agent import UltrafastAgent

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="laya-onnx-ultrafast")
    p.add_argument("--url", default="https://example.com")
    p.add_argument("--goal", required=True)
    p.add_argument("--model", default="receptron/laya-onnx")
    p.add_argument("--providers", default="cpu")
    p.add_argument("--fixture")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--headed", action="store_true")
    p.add_argument("--deterministic", action="store_true")
    args = p.parse_args(argv)
    if args.dry_run and not args.fixture:
        raise SystemExit("--dry-run needs --fixture")
    with UltrafastAgent(args.url, args.goal, model=args.model, providers=args.providers, fixture=args.fixture, headless=not args.headed, deterministic=args.deterministic) as agent:
        for step in agent.run(max_steps=args.max_steps):
            print(json.dumps(step, ensure_ascii=False))
            if step.get("op") in ("DONE", "BLOCKED") or step.get("status") == "max_steps":
                break
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
