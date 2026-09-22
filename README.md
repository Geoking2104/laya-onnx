# laya-onnx

ONNX Runtime for [Laya](https://huggingface.co/convaiinnovations/laya) — typed System-1 decisions, no generated tokens.

PC sibling of [laya-coreml](https://github.com/mizorewww/laya-coreml). Same `choice` / `score` / `noul` contract.

**Play Snake:** https://raw.githack.com/Geoking2104/laya-onnx/main/examples/snake.html

## Onboard

```text
Read https://raw.githubusercontent.com/Geoking2104/laya-onnx/main/ONBOARD.md and follow it end to end.
```

## Ultrafast (NL goal + DOM loop)

Port of [jev-ultrafast](https://github.com/browser-use/jev-ultrafast) / [laya-ultrafast](https://github.com/ipenywis/laya-ultrafast) onto **local ONNX** instead of hosted Jev or MLX.

```bash
pip install -e ".[ultrafast]" && playwright install chromium   # live Chrome
laya-onnx-ultrafast --dry-run --fixture examples/ultrafast_page.json \
  --goal "Find one-way flights from Zurich to London on 20 September 2026."
```

```python
from laya_onnx.ultrafast import UltrafastAgent
with UltrafastAgent(url, goal, model="./onnx", providers="cpu") as agent:
    for step in agent.run():
        print(step["op"], step.get("target"), step["status"])
```

Spec: [docs/ULTRAFAST.md](docs/ULTRAFAST.md)

## Predict

```python
from laya_onnx import load
agent = load("receptron/laya-onnx", providers="cpu")
print(agent.predict("The customer requests a refund.", {"refund": {"type": "noul", "instructions": "Refund?"}}))
```

## Attribution

Apache-2.0. Ultrafast loop design from Browser Use. See LICENSE and NOTICE.
