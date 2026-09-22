"""Offline fixture walk — no Chrome, no Hub weights if you pass a stub agent."""

from laya_onnx.ultrafast import UltrafastAgent

GOAL = "Find one-way flights from Zurich to London on 20 September 2026."

if __name__ == "__main__":
    with UltrafastAgent(
        "https://www.google.com/travel/flights?hl=en",
        GOAL,
        fixture="examples/ultrafast_page.json",
    ) as agent:
        for step in agent.run(max_steps=8):
            print(step["op"], step.get("target"), step["status"])
