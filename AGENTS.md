# laya-onnx — notes for coding agents

Python ≥ 3.11 package. ONNX Runtime backend for Laya typed System-1 decisions (`choice`, `score`, `noul`). No text generation.

- Install: `pip install -e ".[demo]"` (add `openvino` on Intel, `export` for torch→ONNX).
- CLI: `laya-onnx predict|convert|optimize`, `laya-onnx-snake`.
- Browser harness (no weights): `examples/snake.html`.
- CPU Intel: prefer `laya-onnx optimize ./onnx --precision int8`. FP16 is for CUDA.
- Weights are not in git. First `load("receptron/laya-onnx")` hits Hugging Face.
- Do not commit `.venv`, `*.onnx`, `*.onnx.data`.
- Tests: `PYTHONPATH=. pytest -q tests/test_onnx_export.py`.
