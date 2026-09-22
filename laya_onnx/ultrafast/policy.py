from __future__ import annotations
from .ops import OPS
from .snapshot import PageSnapshot

def questions(snap: PageSnapshot, goal: str) -> dict:
    labels = [el.label() for el in snap.elements[:32]] or ["(none)"]
    return {
        "op": {"type": "choice", "instructions": f"Goal: {goal}\nURL: {snap.url}\nTitle: {snap.title}\nPick the next browser operation. Prefer CLICK or TYPE_TEXT. DONE only if the goal is visibly satisfied. BLOCKED if captcha.", "criteria": list(OPS)},
        "ready": {"type": "noul", "instructions": f"Is the goal already satisfied on this page? Goal: {goal}. Visible text: {snap.text[:400]}"},
        "target": {"type": "choice", "instructions": f"Goal: {goal}. Which listed control should the next action use?", "criteria": labels[:16]},
    }

def decide(agent, snap: PageSnapshot, goal: str) -> dict:
    state = f"GOAL\n{goal}\n\nPAGE {snap.url}\n{snap.title}\n{snap.table(40)}"
    out = agent.predict(state, questions(snap, goal))
    answers = out["answers"]
    ready = float(answers.get("ready", {}).get("noul", 0))
    op = answers.get("op", {}).get("choice", "WAIT")
    if ready >= 0.72:
        op = "DONE"
    if op not in OPS:
        op = "WAIT"
    target_label = answers.get("target", {}).get("choice")
    target = None
    if target_label:
        for el in snap.elements:
            if el.label() == target_label:
                target = el.index
                break
        if target is None and snap.elements:
            target = snap.elements[0].index
    return {"op": op, "target": target, "ready": ready, "raw": answers, "usage": out.get("usage", {})}
