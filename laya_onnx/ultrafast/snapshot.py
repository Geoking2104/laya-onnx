from __future__ import annotations
from dataclasses import asdict, dataclass, field

@dataclass
class Element:
    index: int
    role: str
    name: str
    value: str = ""
    tag: str = ""
    editable: bool = False
    def label(self) -> str:
        bits = [f"#{self.index}", self.role or self.tag or "el", self.name[:80]]
        if self.value:
            bits.append(f"= {self.value[:40]}")
        return " ".join(x for x in bits if x)

@dataclass
class PageSnapshot:
    url: str
    title: str = ""
    elements: list = field(default_factory=list)
    text: str = ""
    def table(self, cap: int = 250) -> str:
        return "\n".join(el.label() for el in self.elements[:cap]) or "(no interactive elements)"
    def by_index(self, idx: int):
        for el in self.elements:
            if el.index == idx:
                return el
        return None
    def to_dict(self) -> dict:
        return {"url": self.url, "title": self.title, "text": self.text, "elements": [asdict(e) for e in self.elements]}
    @classmethod
    def from_dict(cls, raw: dict):
        els = [e if isinstance(e, Element) else Element(**e) for e in raw.get("elements", [])]
        return cls(url=raw.get("url", ""), title=raw.get("title", ""), text=raw.get("text", ""), elements=els)

SNAPSHOT_JS = """() => {
  const seen = new WeakSet(); const out = [];
  const nodes = document.querySelectorAll('a,button,input,select,textarea,[role=button],[role=link],[role=textbox],[contenteditable=true]');
  for (const n of nodes) {
    if (seen.has(n) || out.length >= 250) break; seen.add(n);
    const r = n.getBoundingClientRect(); if (r.width < 2 || r.height < 2) continue;
    const role = n.getAttribute('role') || n.tagName.toLowerCase();
    const name = (n.getAttribute('aria-label') || n.getAttribute('placeholder') || n.innerText || n.value || '').trim();
    out.push({index: out.length, role, name: name.slice(0,120), value: String(n.value||'').slice(0,80), tag: n.tagName.toLowerCase(), editable: !!(n.isContentEditable || /input|textarea|select/.test(n.tagName.toLowerCase()))});
  }
  return {url: location.href, title: document.title, text: (document.body.innerText||'').slice(0,1500), elements: out};
}"""
