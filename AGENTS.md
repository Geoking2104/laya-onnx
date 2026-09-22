# laya-onnx — notes for coding agents

Python ≥ 3.11. ONNX Runtime backend for Laya typed System-1 decisions.

- Install: `pip install -e ".[demo]"` (`openvino`, `ultrafast`, `export`).
- CLI: `laya-onnx predict|convert|optimize`, `laya-onnx-snake`, `laya-onnx-ultrafast`.
- Deterministic: `load(..., deterministic=True)` or `--deterministic` (threads=1, pad off, `predict_argmax`, heuristic TYPE_TEXT).
- Snake: https://raw.githack.com/Geoking2104/laya-onnx/main/examples/snake.html
- Ultrafast: `docs/ULTRAFAST.md`, `--fixture examples/ultrafast_page.json`.
- Do not commit `.venv` or `*.onnx`.
