# laya-onnx

ONNX Runtime for [Laya](https://huggingface.co/convaiinnovations/laya) — typed System-1 decisions, no generated tokens.

PC sibling of [laya-coreml](https://github.com/mizorewww/laya-coreml). Same `choice` / `score` / `noul` contract. Weights stay on Hugging Face.

**Play Snake:** https://raw.githack.com/Geoking2104/laya-onnx/main/examples/snake.html

## Onboard

```text
Read https://raw.githubusercontent.com/Geoking2104/laya-onnx/main/ONBOARD.md and follow it end to end.
```

## Install

```bash
git clone https://github.com/Geoking2104/laya-onnx.git && cd laya-onnx
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip && pip install -e ".[demo,dev]"
pip install -e ".[openvino]"   # Intel CPU
pip install -e ".[ultrafast]"  # optional: Playwright DOM loop
```

Model weights are not in git. The first `load("receptron/laya-onnx")` (or the first CLI run) fetches them from the Hub.

## CLI

```bash
laya-onnx predict  --state "..." --questions q.json --model ./onnx
laya-onnx convert  --model convaiinnovations/laya --output onnx --precision fp32
laya-onnx optimize ./onnx --precision int8          # Intel CPU
laya-onnx-ultrafast --dry-run --fixture examples/ultrafast_page.json --goal "..."
laya-onnx-snake --model ./onnx                       # terminal Snake
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

`deterministic=True` sets `threads=1`, `pad_to_multiple=None`, disables the CPU mem arena. Ultrafast then uses `predict_argmax` and heuristic TYPE_TEXT (no TEXT_MODEL). Still not bit-identical across ORT versions — pin `onnxruntime` and the ONNX graph. See [docs/DETERMINISTIC.md](docs/DETERMINISTIC.md).

## Ultrafast (NL goal + DOM loop)

```bash
pip install -e ".[ultrafast]" && playwright install chromium
laya-onnx-ultrafast --dry-run --fixture examples/ultrafast_page.json --goal "..."
```

Spec: [docs/ULTRAFAST.md](docs/ULTRAFAST.md)

## Snake

Terminal demo — real ONNX decisions, guarded by default:

```bash
laya-onnx-snake --model ./onnx                        # arrows/WASD, Space pause, R reset
laya-onnx-snake benchmark --model ./onnx --games 3 --steps 200
laya-onnx-snake --headless --model ./onnx --steps 200 --record run.jsonl
laya-onnx-snake export run.jsonl --html replay.html   # replay/export a recording
```

Browser mock (no weights): https://raw.githack.com/Geoking2104/laya-onnx/main/examples/snake.html — source: [`examples/snake.html`](examples/snake.html).

## Predict

```python
from laya_onnx import load
agent = load("receptron/laya-onnx", providers="cpu")
print(agent.predict("The customer requests a refund.", {"refund": {"type": "noul", "instructions": "Refund?"}}))
```

## Attribution

Apache-2.0. Ultrafast loop design from Browser Use. See LICENSE and NOTICE.
