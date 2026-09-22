# laya-onnx ultrafast

Port of jev-ultrafast / laya-ultrafast onto local ONNX Laya.

`--dry-run` never opens Chrome. `--deterministic` uses `predict_argmax`, ORT threads=1, and heuristic TYPE_TEXT (no TEXT_MODEL).

```bash
laya-onnx-ultrafast --deterministic --dry-run --fixture examples/ultrafast_page.json \
  --goal "Find one-way flights from Zurich to London on 20 September 2026."
```

Ops: CLICK TYPE_TEXT SELECT SCROLL_DOWN SCROLL_UP WAIT DONE BLOCKED
