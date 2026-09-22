# laya-onnx — notes for coding agents

Python ≥ 3.11. ONNX Runtime backend for Laya typed System-1 decisions (`choice`, `score`, `noul`).

- Install: `pip install -e ".[demo]"` (add `openvino` on Intel, `ultrafast` for Playwright, `export` for torch→ONNX).
- CLI: `laya-onnx predict|convert|optimize`, `laya-onnx-snake`, `laya-onnx-ultrafast`.
- Snake: `examples/snake.html` via https://raw.githack.com/Geoking2104/laya-onnx/main/examples/snake.html
- NL-goal DOM loop: `laya_onnx.ultrafast.UltrafastAgent` + `docs/ULTRAFAST.md`. Dry-run: `--fixture examples/ultrafast_page.json`.
- Weights are not in git. First `load("receptron/laya-onnx")` hits Hugging Face.
- Do not commit `.venv`, `*.onnx`.
