# Grok snapshot

This tree is the Grok-maintained ONNX Runtime port of Laya (KayrosLab / Geoking2104).

- Runtime: `laya_onnx/agent.py` (`InferenceSession`, cpu / openvino / cuda)
- Export: `laya_onnx/convert.py` (torch → ONNX opset 17)
- Demo: https://raw.githack.com/Geoking2104/laya-onnx/main/examples/snake.html
- Onboard: `ONBOARD.md`

Do not commit `.venv` or Hub weights.
