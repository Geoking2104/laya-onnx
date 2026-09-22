# Deterministic inference

- `load(..., deterministic=True)` → ORT `intra_op_num_threads=1`, no pad-to-16, CPU arena off.
- `Agent.predict_argmax(state, questions)` → ignore temperature; `choice`/`score`/`noul` from `argmax(logits)`.
- `laya-onnx-ultrafast --deterministic` → `predict_argmax` + `_heuristic_text` (never the OpenAI helper).
- Still not bit-identical across ORT versions. Pin `onnxruntime` and the ONNX graph.
