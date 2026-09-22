import json

import pytest

import laya_onnx.snake.benchmark as benchmark_mod
import laya_onnx.snake.replay as replay_mod
from laya_onnx.snake.game import DIRECTIONS, SnakeGame
from laya_onnx.snake.policy import Decision


class FakePolicy:
    """Offline stand-in for LayaPolicy: always plays the first legal move."""

    def __init__(self, model=None, *, guarded=True, prompt="compact", optimize=False, providers="cpu", threads=None):
        self.guarded = bool(guarded)
        self.metadata = {
            "hardware": "test",
            "model": str(model or "stub"),
            "providers": "cpu",
            "prompt": prompt,
            "optimized": bool(optimize),
        }

    def decide(self, game):
        for direction in DIRECTIONS:
            if game.legal(direction):
                return Decision(direction, direction, 0, {"UP": 1.0}, 0.5)
        return Decision("UP", "UP", 0, {}, 0.5)


def test_benchmark_emits_summary(monkeypatch, capsys):
    monkeypatch.setattr(benchmark_mod, "LayaPolicy", FakePolicy)
    rc = benchmark_mod.main(
        ["--games", "2", "--steps", "12", "--warmup", "1", "--width", "10", "--height", "8"]
    )
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["games"] == 2
    assert summary["steps"] >= 1
    assert summary["p50_inference_ms"] is not None
    assert summary["model"]["hardware"] == "test"


def test_benchmark_rejects_bad_args(monkeypatch):
    monkeypatch.setattr(benchmark_mod, "LayaPolicy", FakePolicy)
    with pytest.raises(SystemExit):
        benchmark_mod.main(["--games", "0"])


def test_replay_summary_and_html(tmp_path, capsys):
    record = tmp_path / "run.jsonl"
    game = SnakeGame(8, 6, 1, 3).snapshot()
    lines = [
        {
            "type": "metadata",
            "format": "laya-onnx-snake-v1",
            "created_utc": "2026-01-01T00:00:00+00:00",
            "model": {"model": "stub"},
            "settings": {},
        },
        {
            "type": "frame",
            "at": 0.0,
            "game": game,
            "decision": {"executed": "UP", "chosen": "UP", "intervened": False, "inference_ms": 1.0},
            "stats": {},
        },
        {"type": "end", "summary": {"steps": 1, "score": 0}, "game": game},
    ]
    record.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")

    html = tmp_path / "replay.html"
    rc = replay_mod.main([str(record), "--html", str(html)])
    assert rc == 0
    assert html.is_file()
    content = html.read_text(encoding="utf-8")
    assert "FRAMES=" in content and '"width": 8' in content

    out = capsys.readouterr().out
    assert "steps" in out and "frames" in out


def test_replay_json_only(tmp_path, capsys):
    record = tmp_path / "run.jsonl"
    lines = [
        {"type": "metadata", "format": "x", "model": {}, "settings": {}},
        {"type": "end", "summary": {"steps": 3}, "game": {}},
    ]
    record.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")
    rc = replay_mod.main([str(record), "--json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out) == {"steps": 3}


def test_replay_missing_file(tmp_path):
    with pytest.raises(SystemExit):
        replay_mod.main([str(tmp_path / "nope.jsonl")])


def test_replay_bad_json(tmp_path):
    record = tmp_path / "bad.jsonl"
    record.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        replay_mod.main([str(record)])
