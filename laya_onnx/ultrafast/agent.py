"""Goal-driven DOM loop. Decisions from local Laya ONNX."""
from __future__ import annotations
import os
from typing import Iterator
from .browser import open_browser
from .ops import MAX_STEPS
from .policy import decide

def _heuristic_text(goal: str, el_name: str) -> str:
    low = (el_name + " " + goal).lower()
    if ("zurich" in goal.lower() or "zürich" in goal.lower()) and ("from" in low or "origin" in low or "depart" in low):
        return "Zurich"
    if "london" in goal.lower() and ("to" in low or "dest" in low):
        return "London"
    return goal.replace(" to ", " | ").split("|")[-1].strip(" ,.")[:40]

def _type_text(goal: str, el_name: str) -> str:
    key = os.environ.get("TEXT_MODEL_API_KEY")
    base = os.environ.get("TEXT_MODEL_BASE_URL")
    if not key and not base:
        return _heuristic_text(goal, el_name)
    try:
        import json, urllib.request
        url = (base or "https://openrouter.ai/api/v1").rstrip("/") + "/chat/completions"
        body = json.dumps({"model": os.environ.get("TEXT_MODEL", "openai/gpt-4.1-mini"), "messages": [{"role": "system", "content": "Return only the exact string to type."}, {"role": "user", "content": f"Goal: {goal}\nField: {el_name}"}], "max_tokens": 32}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", "Authorization": f"Bearer {key or ''}"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode())["choices"][0]["message"]["content"].strip().strip('"')
    except Exception:
        return _heuristic_text(goal, el_name)

class UltrafastAgent:
    def __init__(self, url: str, goal: str, *, model: str = "receptron/laya-onnx", providers: str = "cpu", fixture: str | None = None, headless: bool = True, laya=None, deterministic: bool = False):
        self.url, self.goal, self.fixture, self.headless = url, goal, fixture, headless
        self.deterministic = bool(deterministic)
        if laya is None:
            from laya_onnx import load
            self.laya = load(model, providers=providers, deterministic=self.deterministic)
        else:
            self.laya = laya
        self.browser = None
    def __enter__(self):
        self.browser = open_browser(self.url, fixture=self.fixture, headless=self.headless)
        return self
    def __exit__(self, *exc):
        if self.browser:
            self.browser.close()
        return False
    def run(self, max_steps: int = MAX_STEPS) -> Iterator[dict]:
        if self.browser is None:
            self.browser = open_browser(self.url, fixture=self.fixture, headless=self.headless)
        for step in range(max_steps):
            snap = self.browser.snapshot()
            decision = decide(self.laya, snap, self.goal, argmax=self.deterministic)
            op, target, text, status = decision["op"], decision["target"], None, "ok"
            try:
                if op == "CLICK" and target is not None:
                    self.browser.click(target)
                elif op == "TYPE_TEXT" and target is not None:
                    el = snap.by_index(target)
                    name = el.name if el else ""
                    text = (
                        _heuristic_text(self.goal, name)
                        if self.deterministic
                        else _type_text(self.goal, name)
                    )
                    self.browser.type_text(target, text)
                elif op == "SELECT" and target is not None:
                    self.browser.select(target)
                elif op in ("SCROLL_DOWN", "SCROLL_UP"):
                    self.browser.scroll(op)
                elif op == "WAIT":
                    self.browser.wait(400)
                elif op in ("DONE", "BLOCKED"):
                    yield {"step": step, "op": op, "target": target, "status": op.lower(), "url": snap.url, **decision}
                    return
            except Exception as exc:
                status = f"error:{exc}"
            yield {"step": step, "op": op, "target": target, "text": text, "status": status, "url": snap.url, "ready": decision["ready"]}
        yield {"step": max_steps, "op": "BLOCKED", "status": "max_steps", "url": self.url}
