# laya-onnx

ONNX Runtime port of [laya-mlx](https://github.com/mizorewww/laya-mlx) for [Laya](https://huggingface.co/convaiinnovations/laya) typed System-1 decisions.

Core ML / MLX backends are replaced by `onnxruntime.InferenceSession` (`cpu`, `openvino`, `cuda`). Prompt layout, calibration and the Snake demo stay aligned with upstream Laya / laya-mlx.

## Install

```bash
pip install -e ".[demo]"
pip install -e ".[openvino]"   # Intel
pip install -e ".[export]"     # torch → ONNX
```

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
        }
    },
))
```

```bash
laya-onnx predict --state-file examples/state.json --questions examples/questions.json
laya-onnx convert --model convaiinnovations/laya --output onnx --precision fp32
laya-onnx convert --model convaiinnovations/laya --output onnx-int8 --precision int8
laya-onnx-snake
```

`convert` writes `laya.onnx` + `onnx_config.json` + tokenizer, with dynamic batch/sequence axes, opset 17.

Padding is a multiple of 16. Integer feeds are cast to int64 or int32 to match the graph. A `.mlpackage` path is rejected with an explicit error.

## Layout (from laya-mlx)

| File | Role |
| --- | --- |
| `laya_onnx/agent.py` | ONNX `InferenceSession`, providers, threads, dtype cast |
| `laya_onnx/convert.py` | torch → ONNX + optional INT8 |
| `laya_onnx/inputs.py` | dynamic pad ×16 |
| `laya_onnx/hub.py` | local / HF bundle, `.mlpackage` guard |
| `laya_onnx/cli.py` | `predict` + `convert` |
| `laya_onnx/common.py`, `tokenizer.py`, presets, router, snake | ported from laya-mlx |

Weights stay on Hugging Face. Apache-2.0. See NOTICE.
