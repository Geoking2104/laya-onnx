# laya-onnx ultrafast

Port of [jev-ultrafast](https://github.com/browser-use/jev-ultrafast) / [laya-ultrafast](https://github.com/ipenywis/laya-ultrafast) onto **local ONNX Laya**.

Not a screenshot agent. Each cycle:

1. Snapshot the live page into an indexed element table (`id`, `role`, `name`, `value`).
2. Ask Laya typed questions (`choice` / `noul`) for the next operation and target index.
3. If the op is `TYPE_TEXT`, a small OpenAI-compatible helper writes the string (one call per type, not per click).
4. Execute via Playwright CDP (or a mock page in tests).

Apple-only laya-ultrafast uses MLX. This module uses `laya_onnx.Agent` so Intel / Linux / Windows work.

## API

```python
from laya_onnx.ultrafast import UltrafastAgent

with UltrafastAgent(
    "https://www.google.com/travel/flights?hl=en",
    "Find one-way flights from Zurich to London on 20 September 2026.",
    model="./onnx",
    providers="cpu",
) as agent:
    for step in agent.run(max_steps=40):
        print(step["op"], step.get("target"), step["status"])
```

```bash
laya-onnx-ultrafast --url URL --goal "..." --model ./onnx --providers cpu
laya-onnx-ultrafast --dry-run --fixture examples/ultrafast_page.json --goal "..."
```

`--dry-run` never opens Chrome.

## Operations

`CLICK` `TYPE_TEXT` `SELECT` `SCROLL_DOWN` `SCROLL_UP` `WAIT` `DONE` `BLOCKED`

## Limits

No shadow DOM, frames, canvas, file uploads. Table capped at 250 elements. `DONE` is a model signal.

## Credit

Browser Use (jev-ultrafast) for the indexed action space. ipenywis/laya-ultrafast for the Laya swap.
