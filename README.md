# laya-onnx

ONNX Runtime for [Laya](https://huggingface.co/convaiinnovations/laya) — typed System-1 decisions, no generated tokens.

PC / Linux / Windows / Intel sibling of [laya-coreml](https://github.com/mizorewww/laya-coreml) (Apple Silicon + ANE). Same `choice` / `score` / `noul` contract. Weights stay on Hugging Face.

Playable board (canvas, no model): [examples/snake.html](https://cdn.jsdelivr.net/gh/Geoking2104/laya-onnx@main/examples/snake.html)

---

## Onboard your AI agent

Paste one line into Claude Code, Codex, Cursor or Copilot:

```text
Read https://raw.githubusercontent.com/Geoking2104/laya-onnx/main/ONBOARD.md and follow it end to end.
```

```bash
git clone https://github.com/Geoking2104/laya-onnx.git && cd laya-onnx
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e ".[demo,dev]"
pip install -e ".[openvino]"   # Intel
pip install -e ".[export]"     # torch → ONNX
```

Python ≥ 3.11. Commands: `laya-onnx`, `laya-onnx-snake`.

---

## vs laya-coreml

| | [laya-coreml](https://github.com/mizorewww/laya-coreml) | **laya-onnx** (this repo) |
| --- | --- | --- |
| Target | Apple Silicon, macOS 15+ | PC: Linux / Windows / macOS x86_64 |
| Runtime | Core ML `MLModel` | ONNX Runtime `InferenceSession` |
| Accelerator | ANE + GPU, FP16 / W8 | CPU, OpenVINO, CUDA; INT8 dynamic |
| Published latency | **4.98 ms P50 / 5.31 ms P95** (M3 Max ANE FP16) | measure with `benchmarks/pc_benchmark.py` |
| Energy | 2.78× vs compiled MLX (SMC) | not measured |
| Token budget | ANE L96 or GPU L512–1024 | dynamic pad ×16, `max_len` from bundle |
| Fidelity fixtures | 189/189 FP16, 59/59 ANE | tiny `DecisionModel` parity test only |
| Snake | real Core ML + shield, 49–50 dec/s | CLI `laya-onnx-snake` + canvas HTML mock |
| Convert | `laya-coreml convert` → `.mlpackage` | `laya-onnx convert` → `laya.onnx` |
| Ship | PyPI + Hub `.mlpackage` | git install; Hub ONNX |

Same product idea: one forward pass, calibrated typed answers. Different silicon.

Core ML stays the right choice on M-series. This port is for Intel i7 / 16 GB class machines where ANE does not exist.

---

## Predict

```python
from laya_onnx import load

agent = load("receptron/laya-onnx", providers="cpu", threads=4)
print(agent.predict(
    "The customer requests a refund of a duplicate payment.",
    {"refund": {"type": "noul", "instructions": "Does the customer request a refund?"}},
))
```

```bash
laya-onnx predict --state-file examples/state.json --questions examples/questions.json
```

## Convert & optimize

```bash
laya-onnx convert --model convaiinnovations/laya --output onnx --precision fp32
laya-onnx optimize ./onnx --precision int8
```

Intel CPU: INT8. CUDA: FP16. A `.mlpackage` path is rejected — use laya-coreml there.

## Snake

```bash
laya-onnx-snake --model ./onnx
```

Browser mock: https://cdn.jsdelivr.net/gh/Geoking2104/laya-onnx@main/examples/snake.html

Do not use `htmlpreview.github.io` — it often skips JavaScript.

## Benchmark

```bash
PYTHONPATH=. python benchmarks/pc_benchmark.py ./onnx --calls 200 --providers cpu
```

## Attribution

Apache-2.0. Independent ONNX port. Upstream: Laya, laya-mlx, [laya-coreml](https://github.com/mizorewww/laya-coreml). Not an official Convai or Apple release. See LICENSE and NOTICE.
