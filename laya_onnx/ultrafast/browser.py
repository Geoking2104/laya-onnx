"""Playwright driver with a fixture/mock fallback."""
from __future__ import annotations
import json
from pathlib import Path
from .snapshot import PageSnapshot, SNAPSHOT_JS

class MockBrowser:
    def __init__(self, fixture):
        self.snap = fixture if isinstance(fixture, PageSnapshot) else PageSnapshot.from_dict(fixture)
        self.actions = []
    def snapshot(self):
        return self.snap
    def click(self, index: int):
        self.actions.append({"op": "CLICK", "target": index})
    def type_text(self, index: int, text: str):
        el = self.snap.by_index(index)
        if el:
            el.value = text
        self.actions.append({"op": "TYPE_TEXT", "target": index, "text": text})
    def select(self, index: int):
        self.actions.append({"op": "SELECT", "target": index})
    def scroll(self, direction: str):
        self.actions.append({"op": "SCROLL", "direction": direction})
    def wait(self, ms: int = 400):
        self.actions.append({"op": "WAIT", "ms": ms})
    def close(self):
        pass

class PlaywrightBrowser:
    def __init__(self, url: str, headless: bool = True):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=headless)
        self._page = self._browser.new_page()
        self._page.goto(url, wait_until="domcontentloaded", timeout=30000)
    def snapshot(self):
        return PageSnapshot.from_dict(self._page.evaluate(SNAPSHOT_JS))
    def click(self, index: int):
        self._page.evaluate("(i) => { const n = document.querySelectorAll('a,button,input,select,textarea,[role=button],[role=link]')[i]; n && n.click(); }", int(index))
    def type_text(self, index: int, text: str):
        self._page.evaluate("([i, t]) => { const n = document.querySelectorAll('input,textarea,select,[role=textbox]')[i]; if (n) { n.focus(); n.value = t; n.dispatchEvent(new Event('input', {bubbles:true})); } }", [int(index), text])
    def select(self, index: int):
        self.click(index)
    def scroll(self, direction: str):
        self._page.mouse.wheel(0, 600 if direction == "SCROLL_DOWN" else -600)
    def wait(self, ms: int = 400):
        self._page.wait_for_timeout(ms)
    def close(self):
        self._browser.close(); self._pw.stop()

def open_browser(url: str, *, fixture=None, headless: bool = True):
    if fixture:
        return MockBrowser(json.loads(Path(fixture).read_text()))
    try:
        return PlaywrightBrowser(url, headless=headless)
    except Exception as exc:
        raise RuntimeError("Playwright missing. pip install -e '.[ultrafast]' && playwright install chromium, or pass --fixture") from exc
