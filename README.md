# laya-onnx

ONNX Runtime for [Laya](https://huggingface.co/convaiinnovations/laya) — typed System-1 decisions, no text generation.

Port of [laya-mlx](https://github.com/mizorewww/laya-mlx) / Core ML to PC (`cpu`, `openvino`, `cuda`). One forward pass returns calibrated `choice` / `score` / `noul`. Weights stay on Hugging Face (~0.8–1.7 GB).

---

## Onboard your AI agent

Paste **one line** into Claude Code, Codex, Cursor, Copilot, or any coding agent. It clones the repo, creates the venv, installs the CLI, and writes workspace notes.

```text
Read https://raw.githubusercontent.com/Geoking2104/laya-onnx/main/ONBOARD.md and follow it end to end.
```

Works with Claude Code, Codex, Cursor, GitHub Copilot, Windsurf, Cline, and similar agents. Safe to re-run. No RunPod account required for CPU.

Human equivalent:

```bash
git clone https://github.com/Geoking2104/laya-onnx.git
cd laya-onnx
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e ".[demo,dev]"
# Intel:
pip install -e ".[openvino]"
# torch → ONNX:
pip install -e ".[export]"
```

Python ≥ 3.11. Entry points: `laya-onnx`, `laya-onnx-snake`.

GPU cloud (optional, not this package):

```bash
npx skills add runpod/runpod-plugins-official
```

---

## Predict

```python
from laya_onnx import load

agent = load("receptron/laya-onnx", providers="cpu", threads=4)
print(agent.predict(
    "I was billed twice. Please refund the duplicate.",
    {
        "department": {
            "type": "choice",
            "instructions": "Who should handle this?",
            "criteria": ["billing", "technical", "sales"],
        },
        "refund": {
            "type": "noul",
            "instructions": "Does the customer ask for money back?",
        },
    },
))
```

```bash
laya-onnx predict --state-file examples/state.json --questions examples/questions.json --providers cpu
```

## Convert & optimize

```bash
laya-onnx convert --model convaiinnovations/laya --output onnx --precision fp32
laya-onnx optimize ./onnx --precision int8
```

On Intel CPU use **int8**. FP16 is for CUDA.

## Snake

```bash
laya-onnx-snake --model ./onnx
```

Browser harness (no weights): [`examples/snake.html`](examples/snake.html).

## Benchmark

```bash
PYTHONPATH=. python benchmarks/pc_benchmark.py ./onnx --calls 200 --providers cpu
```

## Tests

```bash
PYTHONPATH=. pytest -q tests/test_onnx_export.py
```

## Layout

| Path | Role |
| --- | --- |
| `ONBOARD.md` | one-shot prompt for coding agents |
| `AGENTS.md` | repo conventions |
| `laya_onnx/agent.py` | `predict` |
| `laya_onnx/optimize.py` | INT8 / FP16 |
| `laya_onnx/snake/cli.py` | live Snake |
| `examples/snake.html` | embedded test |

## Attribution

Apache-2.0. Weights: [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya). ONNX: [receptron/laya-onnx](https://huggingface.co/receptron/laya-onnx). See LICENSE and NOTICE.
