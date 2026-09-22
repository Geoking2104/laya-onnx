# laya-onnx

ONNX Runtime for [Laya](https://huggingface.co/convaiinnovations/laya) — typed System-1 decisions, no generated tokens.

PC / Linux / Windows / Intel sibling of [laya-coreml](https://github.com/mizorewww/laya-coreml) (Apple Silicon + ANE). Same `choice` / `score` / `noul` contract. Weights stay on Hugging Face.

Playable board (English UI, canvas, performance dashboard):
https://cdn.jsdelivr.net/gh/Geoking2104/laya-onnx@main/examples/snake.html

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

```bash
laya-onnx-snake --model ./onnx
```

Browser mock: https://cdn.jsdelivr.net/gh/Geoking2104/laya-onnx@main/examples/snake.html

Keyboard mode shows a shortcut panel (↑↓←→ / WASD, Space, R). Performance panel: score, best, length, steps, eats, deaths, elapsed, steps/s, length sparkline.

Use jsDelivr, not htmlpreview (JS is often blocked there).

## Convert

```bash
laya-onnx convert --model convaiinnovations/laya --output onnx --precision fp32
laya-onnx optimize ./onnx --precision int8
```

## Attribution

Apache-2.0. See LICENSE and NOTICE.
