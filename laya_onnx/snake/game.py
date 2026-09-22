"""Deterministic, JSON-serializable Snake game used by the terminal demo.

No third-party imports: the game must be runnable (and testable) with an
empty environment. The board is a ``width`` x ``height`` grid with the head
first in ``body``. ``seed`` fully determines the food sequence.
"""

from __future__ import annotations

import random

DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
DELTAS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
MIN_SIZE = 4


class SnakeGame:
    def __init__(self, width: int = 24, height: int = 16, seed: int = 7, initial_length: int = 6):
        if not isinstance(width, int) or not isinstance(height, int) or isinstance(width, bool) or isinstance(height, bool):
            raise ValueError("width and height must be integers")
        if width < MIN_SIZE or height < MIN_SIZE:
            raise ValueError(f"width and height must be at least {MIN_SIZE}")
        area = width * height
        if (
            not isinstance(initial_length, int)
            or isinstance(initial_length, bool)
            or not 2 <= initial_length < area
        ):
            raise ValueError(f"initial_length must be an integer between 2 and {area - 1}")
        self.width = width
        self.height = height
        self.seed = seed
        self.rng = random.Random(seed)
        path = self._cycle()
        start = path.index((width // 2, height // 2))
        span = len(path)
        # Head sits at the centre; the body trails backwards along the path.
        self.body = [path[(start - i) % span] for i in range(initial_length)]
        self.steps = 0
        self.score = 0
        self.alive = True
        self.won = False
        self.food = self._spawn()

    # -- internals ---------------------------------------------------------
    def _cycle(self):
        """Boustrophedon Hamiltonian path over the grid (row by row)."""
        path = []
        for y in range(self.height):
            xs = range(self.width) if y % 2 == 0 else range(self.width - 1, -1, -1)
            for x in xs:
                path.append((x, y))
        return path

    def _spawn(self):
        occupied = set(self.body)
        free = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in occupied
        ]
        if not free:
            return None
        return self.rng.choice(free)

    def _target(self, direction):
        dx, dy = DELTAS[direction]
        hx, hy = self.body[0]
        return hx + dx, hy + dy

    # -- public API --------------------------------------------------------
    def legal(self, direction) -> bool:
        """True if ``direction`` can be played without dying. Pure."""
        if not self.alive or self.won or direction not in DELTAS:
            return False
        nx, ny = self._target(direction)
        if not (0 <= nx < self.width and 0 <= ny < self.height):
            return False
        grows = self.food is not None and (nx, ny) == self.food
        # Without growth the tail cell vacates, so stepping onto it is safe.
        blocked = set(self.body if grows else self.body[:-1])
        return (nx, ny) not in blocked

    def step(self, direction):
        """Advance one tick. Returns ``ok`` / ``eat`` / ``dead`` / ``over``."""
        if direction not in DELTAS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
        if not self.alive:
            return "dead"
        if self.won:
            return "over"
        if not self.legal(direction):
            self.alive = False
            return "dead"
        nx, ny = self._target(direction)
        eats = self.food is not None and (nx, ny) == self.food
        self.body.insert(0, (nx, ny))
        if eats:
            self.score += 1
            self.food = self._spawn()
        else:
            self.body.pop()
        self.steps += 1
        if self.food is None or len(self.body) >= self.width * self.height:
            self.won = True
        return "eat" if eats else "ok"

    def snapshot(self) -> dict:
        """JSON-serializable view of the current state."""
        return {
            "width": self.width,
            "height": self.height,
            "head": list(self.body[0]),
            "body": [list(cell) for cell in self.body],
            "food": list(self.food) if self.food is not None else None,
            "score": self.score,
            "length": len(self.body),
            "steps": self.steps,
            "alive": self.alive,
            "won": self.won,
        }
