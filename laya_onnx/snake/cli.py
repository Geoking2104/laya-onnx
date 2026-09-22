"""Run the live terminal demo. Also supports benchmark and export subcommands."""

import argparse
import json
import sys
import time
from collections import deque
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.live import Live

from .game import SnakeGame
from .policy import LayaPolicy
from .ui import BG, compose, layout_size


class Keyboard:
    """Non-blocking raw-key reader: cbreak on POSIX, msvcrt on Windows."""

    def __enter__(self):
        self._posix = sys.platform != "win32"
        self.saved = None
        if sys.stdin.isatty():
            if self._posix:
                import termios
                import tty

                self.fd = sys.stdin.fileno()
                self.saved = termios.tcgetattr(self.fd)
                tty.setcbreak(self.fd)
            else:
                import msvcrt  # noqa: F401  (present on Windows)

                self.saved = True
        return self

    def read(self):
        if not self.saved:
            return ""
        if self._posix:
            import os
            import select

            if select.select([sys.stdin], [], [], 0)[0]:
                return os.read(self.fd, 128).decode(errors="ignore")
            return ""
        import msvcrt

        chars = []
        while msvcrt.kbhit():
            chars.append(msvcrt.getwch())
        return "".join(chars)

    def __exit__(self, *_):
        if self._posix and self.saved:
            import termios

            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)


def positive(value):
    result = float(value)
    if not 0 < result < float("inf"):
        raise argparse.ArgumentTypeError("Expected a positive finite number")
    return result


def play(argv=None):
    parser = argparse.ArgumentParser(
        prog="laya-onnx-snake",
        description=__doc__,
        epilog="Also: laya-onnx-snake benchmark --help | laya-onnx-snake export --help",
    )
    parser.add_argument("--model", help="Local model directory or an already cached Hub ID")
    parser.add_argument("--prompt", choices=("compact", "detailed"), default="compact")
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable 16-token buckets",
    )
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--initial-length", type=int, default=6)
    parser.add_argument("--fps", type=positive, default=12)
    parser.add_argument("--max-speed", action="store_true")
    parser.add_argument("--duration", type=positive)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--unassisted", action="store_true")
    parser.add_argument("--record", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-alt-screen", action="store_true")
    args = parser.parse_args(argv)
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    try:
        game = SnakeGame(args.width, args.height, args.seed, args.initial_length)
    except ValueError as error:
        parser.error(str(error))
    console = Console(style=f"on {BG}", highlight=False)
    if not args.headless and not console.is_terminal:
        parser.error("Interactive display needs a TTY. Use --headless.")
    print("Loading local ONNX bundle; Hub only if the path is a repo id...", file=sys.stderr)
    policy = LayaPolicy(
        args.model, guarded=not args.unassisted, prompt=args.prompt, optimize=args.optimize
    )
    warm = SnakeGame(args.width, args.height, args.seed + 10000, args.initial_length)
    for _ in range(6):
        decision = policy.decide(warm)
        warm.step(decision.executed)
        if not warm.alive:
            break
    record = None
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        record = args.record.open("x")
        record.write(
            json.dumps(
                {
                    "type": "metadata",
                    "format": "laya-onnx-snake-v1",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "model": policy.metadata,
                    "settings": {
                        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                    },
                }
            )
            + "\n"
        )
    started = time.perf_counter()
    stats = {
        "hardware": policy.metadata["hardware"],
        "guarded": policy.guarded,
        "interventions": 0,
        "best": 0,
        "round": 1,
        "paused": False,
        "elapsed": 0,
        "steps_per_second": 0,
    }
    calls = total_steps = deaths = 0
    timestamps = deque(maxlen=60)
    inference = []
    displayed_board, displayed_decision = game.snapshot(), {}
    live = (
        Live(console=console, screen=not args.no_alt_screen, auto_refresh=False, vertical_overflow="crop")
        if not args.headless
        else None
    )
    try:
        with Keyboard() as keys, live if live else nullcontext():
            while True:
                now = time.perf_counter()
                if (args.duration and now - started >= args.duration) or (
                    args.steps and total_steps >= args.steps
                ):
                    break
                pressed = keys.read().lower()
                if "q" in pressed or "\x03" in pressed:
                    break
                if " " in pressed:
                    stats["paused"] = not stats["paused"]
                if "+" in pressed:
                    args.fps = min(240, args.fps + 2)
                if "-" in pressed:
                    args.fps = max(1, args.fps - 2)
                if "r" in pressed:
                    stats["round"] += 1
                    game = SnakeGame(args.width, args.height, args.seed + stats["round"] - 1, args.initial_length)
                    displayed_board, displayed_decision = game.snapshot(), {}
                if stats["paused"]:
                    stats["elapsed"] = now - started
                    if live:
                        live.update(compose(displayed_board, displayed_decision, stats).rich_text(), refresh=True)
                    time.sleep(0.03)
                    continue
                minimum_width, minimum_height = layout_size(game.width, game.height)
                if live and (console.width < minimum_width or console.height < minimum_height):
                    live.update(
                        f"Resize terminal to at least {minimum_width}x{minimum_height}. Q quits.",
                        refresh=True,
                    )
                    time.sleep(0.1)
                    continue
                decision = policy.decide(game)
                calls += 1
                inference.append(decision.inference_ms)
                stats["interventions"] += decision.intervened
                shown = time.perf_counter()
                timestamps.append(shown)
                stats["elapsed"] = shown - started
                stats["steps_per_second"] = (
                    (len(timestamps) - 1) / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0
                )
                stats["best"] = max(stats["best"], game.score)
                board = game.snapshot()
                displayed_board, displayed_decision = board, decision.to_dict()
                if live:
                    live.update(compose(board, decision.to_dict(), stats).rich_text(), refresh=True)
                if record:
                    record.write(
                        json.dumps(
                            {"type": "frame", "at": shown - started, "game": board, "decision": decision.to_dict(), "stats": dict(stats)},
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                if not args.max_speed:
                    remaining = 1 / args.fps - (time.perf_counter() - now)
                    if remaining > 0:
                        time.sleep(remaining)
                game.step(decision.executed)
                total_steps += 1
                stats["best"] = max(stats["best"], game.score)
                if not game.alive or game.won:
                    deaths += not game.alive
                    if args.unassisted:
                        break
                    if live:
                        live.update(compose(game.snapshot(), {}, stats).rich_text(), refresh=True)
                        time.sleep(1)
                    stats["round"] += 1
                    game = SnakeGame(args.width, args.height, args.seed + stats["round"] - 1, args.initial_length)
    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.perf_counter() - started
        summary = {
            "steps": total_steps,
            "inference_calls": calls,
            "seconds": elapsed,
            "steps_per_second": total_steps / elapsed if elapsed else 0,
            "score": game.score,
            "length": len(game.body),
            "best_score": stats["best"],
            "interventions": stats["interventions"],
            "deaths": deaths,
            "guarded": policy.guarded,
            "network": "offline",
            "mean_inference_ms": sum(inference) / len(inference) if inference else None,
        }
        if record:
            record.write(json.dumps({"type": "end", "summary": summary, "game": game.snapshot()}) + "\n")
            record.close()
        print(json.dumps(summary, indent=2))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "benchmark":
        from .benchmark import main as benchmark

        return benchmark(argv[1:])
    if argv and argv[0] == "export":
        from .replay import main as export

        return export(argv[1:])
    return play(argv)
