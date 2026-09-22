"""Headless throughput benchmark for the Snake policy (no display, no TTY)."""

from __future__ import annotations

import argparse
import json
import statistics
import time

from .game import SnakeGame
from .policy import LayaPolicy


def _percentile(sorted_samples, q):
    if not sorted_samples:
        return None
    index = min(len(sorted_samples) - 1, int(len(sorted_samples) * q))
    return sorted_samples[index]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="laya-onnx-snake benchmark", description=__doc__)
    parser.add_argument("--model", help="Local model directory or an already cached Hub ID")
    parser.add_argument("--prompt", choices=("compact", "detailed"), default="compact")
    parser.add_argument("--optimize", action="store_true", help="Enable 16-token buckets")
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--steps", type=int, default=200, help="Max steps per game")
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--height", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--initial-length", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=6, help="Warm-up decisions per game")
    parser.add_argument("--unassisted", action="store_true")
    args = parser.parse_args(argv)
    if args.games < 1:
        parser.error("--games must be positive")
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

    policy = LayaPolicy(
        args.model, guarded=not args.unassisted, prompt=args.prompt, optimize=args.optimize
    )
    samples = []
    total_steps = deaths = wins = interventions = best = 0
    started = time.perf_counter()
    for game_index in range(args.games):
        game = SnakeGame(args.width, args.height, args.seed + game_index, args.initial_length)
        for _ in range(args.warmup):
            if not game.alive or game.won:
                break
            game.step(policy.decide(game).executed)
        played = 0
        while played < args.steps and game.alive and not game.won:
            decision = policy.decide(game)
            samples.append(decision.inference_ms)
            interventions += decision.intervened
            game.step(decision.executed)
            played += 1
            total_steps += 1
        best = max(best, game.score)
        if game.won:
            wins += 1
        elif not game.alive:
            deaths += 1
    seconds = time.perf_counter() - started
    samples.sort()
    summary = {
        "model": policy.metadata,
        "games": args.games,
        "steps": total_steps,
        "seconds": round(seconds, 3),
        "steps_per_second": round(total_steps / seconds, 2) if seconds else 0,
        "best_score": best,
        "wins": wins,
        "deaths": deaths,
        "interventions": interventions,
        "mean_inference_ms": round(statistics.fmean(samples), 3) if samples else None,
        "p50_inference_ms": round(_percentile(samples, 0.50), 3) if samples else None,
        "p95_inference_ms": round(_percentile(samples, 0.95), 3) if samples else None,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
