"""Rich rendering for the terminal Snake demo.

``compose`` returns a small wrapper whose ``rich_text()`` yields a
``rich.text.Text`` renderable, matching how ``snake/cli.py`` drives
``rich.live.Live``. Kept import-light so it can be unit-tested headlessly.
"""

from __future__ import annotations

from rich.text import Text

BG = "black"

HEAD_GLYPH = "█"
BODY_GLYPH = "▪"
FOOD_GLYPH = "◆"
EMPTY_GLYPH = "·"

HEAD_STYLE = "bold bright_green"
BODY_STYLE = "green"
FOOD_STYLE = "bold yellow"
EMPTY_STYLE = "grey35"


def layout_size(width: int, height: int) -> tuple[int, int]:
    """Minimum console (columns, rows) for a board of ``width`` x ``height``."""
    columns = max(2 * int(width) + 2, 48)
    rows = int(height) + 4
    return columns, rows


def _cells(board: dict):
    width = int(board.get("width") or 0)
    height = int(board.get("height") or 0)
    body = [tuple(cell) for cell in (board.get("body") or [])]
    head = tuple(board["head"]) if board.get("head") else (body[0] if body else None)
    food = tuple(board["food"]) if board.get("food") else None
    occupied = set(body)
    grid = []
    for y in range(height):
        row = []
        for x in range(width):
            point = (x, y)
            if head is not None and point == head:
                row.append((HEAD_GLYPH, HEAD_STYLE))
            elif point in occupied:
                row.append((BODY_GLYPH, BODY_STYLE))
            elif food is not None and point == food:
                row.append((FOOD_GLYPH, FOOD_STYLE))
            else:
                row.append((EMPTY_GLYPH, EMPTY_STYLE))
        grid.append(row)
    return grid


class Frame:
    def __init__(self, board: dict, decision: dict, stats: dict):
        self.board = board or {}
        self.decision = decision or {}
        self.stats = stats or {}

    def rich_text(self) -> Text:
        text = Text(style=f"on {BG}")
        for y, row in enumerate(_cells(self.board)):
            if y:
                text.append("\n")
            for x, (glyph, style) in enumerate(row):
                text.append(glyph, style=style)
                if x < len(row) - 1:
                    text.append(" ")
        board = self.board
        stats = self.stats
        decision = self.decision
        text.append("\n\n")
        text.append(
            "score {score}  best {best}  len {length}  steps {steps}  "
            "{sps:.1f}/s  guarded {guarded}  int {interventions}".format(
                score=board.get("score", 0),
                best=stats.get("best", 0),
                length=board.get("length", 0),
                steps=board.get("steps", 0),
                sps=float(stats.get("steps_per_second", 0.0) or 0.0),
                guarded=stats.get("guarded", "-"),
                interventions=stats.get("interventions", 0),
            ),
            style="dim",
        )
        text.append("\n")
        guarded = "  [guarded]" if decision.get("intervened") else ""
        text.append(
            "next {executed}  model {chosen}{guarded}  {ms:.2f} ms".format(
                executed=decision.get("executed", "-"),
                chosen=decision.get("chosen", "-"),
                guarded=guarded,
                ms=float(decision.get("inference_ms", 0.0) or 0.0),
            ),
            style="cyan",
        )
        return text


def compose(board: dict, decision: dict, stats: dict) -> Frame:
    return Frame(board, decision, stats)
