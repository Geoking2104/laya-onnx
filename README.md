# laya-onnx

ONNX Runtime for [Laya](https://huggingface.co/convaiinnovations/laya) — typed System-1 decisions, no generated tokens.

PC / Linux / Windows / Intel sibling of [laya-coreml](https://github.com/mizorewww/laya-coreml) (Apple Silicon + ANE). Same `choice` / `score` / `noul` contract. Weights stay on Hugging Face.

**Play Snake (working preview):**
https://raw.githack.com/Geoking2104/laya-onnx/main/examples/snake.html

English UI, 720×480 canvas, keyboard map, performance dashboard.

---

## Onboard your AI agent

```text
Read https://raw.githubusercontent.com/Geoking2104/laya-onnx/main/ONBOARD.md and follow it end to end.
```

```bash
git clone https://github.com/Geoking2104/laya-onnx.git && cd laya-onnx
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e ".[demo,dev]"
pip install -e ".[openvino]"
pip install -e ".[export]"
```

---

## vs laya-coreml

| | [laya-coreml](https://github.com/mizorewww/laya-coreml) | **laya-onnx** |
| --- | --- | --- |
| Target | Apple Silicon | PC / Intel |
| Runtime | Core ML | ONNX Runtime |
| Accelerator | ANE FP16 / W8 | CPU, OpenVINO, CUDA, INT8 |
| Published latency | 4.98 / 5.31 ms P50/P95 | measure locally |
| Snake | real Core ML | CLI + canvas mock |

---

## Predict

```python
from laya_onnx import load
agent = load("receptron/laya-onnx", providers="cpu", threads=4)
print(agent.predict("The customer requests a refund of a duplicate payment.", {"refund": {"type": "noul", "instructions": "Does the customer request a refund?"}}))
```

## Snake

Terminal (real ONNX weights):

```bash
laya-onnx-snake --model ./onnx
```

Browser mock (no weights):

- **Preview that runs JS:** https://raw.githack.com/Geoking2104/laya-onnx/main/examples/snake.html
- Source: [`examples/snake.html`](examples/snake.html)

Keyboard: arrows or WASD, Space pause, R reset. Dashboard: score, best, length, steps, eats, deaths, elapsed, steps/s, length sparkline.

Do not use `htmlpreview.github.io` (often skips scripts). jsDelivr `@main` can lag behind this repo.

## Convert

```bash
laya-onnx convert --model convaiinnovations/laya --output onnx --precision fp32
laya-onnx optimize ./onnx --precision int8
```

## Attribution

Apache-2.0. See LICENSE and NOTICE.
