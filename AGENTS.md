# laya-onnx — notes for coding agents

Python ≥ 3.11. ONNX Runtime backend for Laya typed System-1 decisions.

- `load(..., deterministic=True)` and `predict_argmax` for replayable tools.
- CLI: `laya-onnx`, `laya-onnx-snake`, `laya-onnx-ultrafast --deterministic`.
- Specs: `docs/ULTRAFAST.md`, `docs/DETERMINISTIC.md`.
- Do not commit `.venv` or `*.onnx`.
