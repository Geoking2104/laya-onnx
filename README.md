# laya-onnx

ONNX Runtime for [Laya](https://huggingface.co/convaiinnovations/laya) — typed System-1 decisions, no generated tokens.

PC sibling of [laya-coreml](https://github.com/mizorewww/laya-coreml). Same `choice` / `score` / `noul` contract.

**Play Snake:** https://raw.githack.com/Geoking2104/laya-onnx/main/examples/snake.html

## Onboard

```text
Read https://raw.githubusercontent.com/Geoking2104/laya-onnx/main/ONBOARD.md and follow it end to end.
```

## Deterministic mode

`predict()` already uses argmax. Residual jitter comes from ORT threads, dynamic pad-to-16, temperature on confidence, and the optional TYPE_TEXT LLM.

```python
from laya_onnx import load
agent = load("./onnx", providers="cpu", deterministic=True)
print(agent.predict_argmax(state, questions))  # ignore temperature
```

```bash
laya-onnx predict --deterministic --state "..." --questions q.json --model ./onnx
laya-onnx-ultrafast --deterministic --dry-run --fixture examples/ultrafast_page.json \
  --goal "Find one-way flights from Zurich to London on 20 September 2026."
```

`deterministic=True` sets `threads=1`, `pad_to_multiple=None`, disables the CPU mem arena. Ultrafast then uses heuristic TYPE_TEXT (no TEXT_MODEL). Replay logits in CI if two ORT builds must match bit-exactly.

## Ultrafast (NL goal + DOM loop)

```bash
pip install -e ".[ultrafast]" && playwright install chromium
laya-onnx-ultrafast --dry-run --fixture examples/ultrafast_page.json --goal "..."
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
