"""Laya-backed Snake policy.

Each tick builds a typed System-1 decision: the local ONNX Agent is asked
(a ``choice`` over UP/DOWN/LEFT/RIGHT) and its answer becomes the move. In
``guarded`` mode an illegal move is replaced by the best legal alternative
and the tick is counted as an intervention.
"""

from __future__ import annotations

import os
import platform
import time

from .. import load
from .game import DIRECTIONS


def _hardware() -> str:
    chip = platform.processor() or platform.machine() or "cpu"
    return f"{platform.system()} {chip} x{os.cpu_count() or 1}"


class Decision:
    def __init__(self, chosen, executed, intervened, probabilities, inference_ms, confidence=None):
        self.chosen = chosen
        self.executed = executed
        self.intervened = int(intervened)
        self.probabilities = probabilities
        self.inference_ms = float(inference_ms)
        self.confidence = confidence

    def to_dict(self) -> dict:
        payload = {
            "chosen": self.chosen,
            "executed": self.executed,
            "intervened": bool(self.intervened),
            "probabilities": self.probabilities,
            "inference_ms": round(self.inference_ms, 3),
        }
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        return payload


class LayaPolicy:
    def __init__(
        self,
        model=None,
        *,
        guarded: bool = True,
        prompt: str = "compact",
        optimize: bool = False,
        providers: str = "cpu",
        threads=None,
    ):
        if prompt not in ("compact", "detailed"):
            raise ValueError("prompt must be 'compact' or 'detailed'")
        self.model_id = str(model or "receptron/laya-onnx")
        self.guarded = bool(guarded)
        self.prompt = prompt
        self.optimize = bool(optimize)
        # "optimize" keeps 16-token padding buckets; otherwise skip padding for latency.
        self.agent = load(
            model or "receptron/laya-onnx",
            providers=providers,
            threads=threads,
            pad_to_multiple=16 if self.optimize else None,
        )
        self.metadata = {
            "hardware": _hardware(),
            "model": self.model_id,
            "providers": self.agent.providers,
            "prompt": self.prompt,
            "optimized": self.optimize,
        }

    # -- prompt building ---------------------------------------------------
    def _state(self, game) -> str:
        snap = game.snapshot()
        head, food = snap["head"], snap["food"]
        if self.prompt == "detailed":
            rows = []
            body = {tuple(cell) for cell in snap["body"]}
            for y in range(snap["height"]):
                line = []
                for x in range(snap["width"]):
                    point = (x, y)
                    if point == tuple(head):
                        line.append("#")
                    elif food is not None and point == tuple(food):
                        line.append("*")
                    elif point in body:
                        line.append("o")
                    else:
                        line.append(".")
                rows.append("".join(line))
            grid = "\n".join(rows)
            return (
                f"Snake {snap['width']}x{snap['height']}\n{grid}\n"
                f"#=head o=body *=food head={head} food={food} "
                f"length={snap['length']} score={snap['score']}"
            )
        return (
            f"Snake {snap['width']}x{snap['height']} head={head} food={food} "
            f"length={snap['length']} score={snap['score']} alive={snap['alive']}"
        )

    def _questions(self) -> dict:
        return {
            "move": {
                "type": "choice",
                "instructions": (
                    "Pick the snake's next move. Move toward the food and never "
                    "hit the wall or the snake's own body."
                ),
                "criteria": {direction: direction.lower() for direction in DIRECTIONS},
            }
        }

    # -- decision ----------------------------------------------------------
    def decide(self, game) -> Decision:
        started = time.perf_counter()
        out = self.agent.predict(self._state(game), self._questions())
        inference_ms = (time.perf_counter() - started) * 1000.0
        answer = (out.get("answers") or {}).get("move") or {}
        probabilities = answer.get("probabilities") or {}
        ranked = sorted(
            DIRECTIONS, key=lambda d: float(probabilities.get(d, 0.0)), reverse=True
        )
        chosen = answer.get("choice")
        if chosen not in DIRECTIONS:
            chosen = ranked[0]
        executed, intervened = chosen, 0
        if self.guarded and not game.legal(chosen):
            fallback = next((d for d in ranked if game.legal(d)), None)
            if fallback is None:
                fallback = next((d for d in DIRECTIONS if game.legal(d)), chosen)
            executed, intervened = fallback, 1
        return Decision(
            chosen, executed, intervened, probabilities, inference_ms, answer.get("confidence")
        )
