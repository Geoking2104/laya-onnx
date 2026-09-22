# laya-onnx

Runtime **ONNX Runtime** pour [Laya](https://huggingface.co/convaiinnovations/laya) — décisions typées System-1, sans génération de texte.

Portage de [laya-mlx](https://github.com/mizorewww/laya-mlx) / laya-coreml vers PC : Core ML et MLX sont remplacés par `onnxruntime.InferenceSession` (`cpu`, `openvino`, `cuda`). Le layout de prompt, la calibration et la démo Snake restent alignés sur Laya amont.

Laya prend un **état** (texte, email, ticket, JSON) et des **questions typées** (`choice`, `score`, `noul`) et rend, en **une seule passe**, des distributions calibrées. Rien n’est échantillonné.

Les poids (~0.8–1.7 Go) ne sont **pas** dans ce dépôt. Ils se téléchargent depuis Hugging Face au premier usage.

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[demo]"
pip install -e ".[openvino]"   # Intel OpenVINO
pip install -e ".[gpu]"        # CUDA
pip install -e ".[export]"     # torch → ONNX
```

Python ≥ 3.11. Scripts installés : `laya-onnx`, `laya-onnx-snake`.

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
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["not urgent", "soon", "blocking"],
        },
        "refund": {
            "type": "noul",
            "instructions": "Does the customer ask for money back?",
        },
    },
))
```

CLI :

```bash
laya-onnx predict \
  --state-file examples/state.json \
  --questions examples/questions.json \
  --providers cpu \
  --threads 4
```

`Agent` / `load` / `predict` (`system_one`) : même contrat que laya-mlx.

| Type | Sortie |
| --- | --- |
| `choice` | `choice` + `probabilities` par label |
| `score` | `score` (espérance 0..K-1) + `legend` |
| `noul` | `noul` = P(true) |

## Convert & optimize

```bash
laya-onnx convert --model convaiinnovations/laya --output onnx --precision fp32
laya-onnx convert --model convaiinnovations/laya --output onnx-int8 --precision int8

laya-onnx optimize ./onnx --precision int8
laya-onnx optimize ./onnx --output onnx-fp16 --precision fp16
```

`convert` écrit `laya.onnx` (+ `.data` éventuel), `onnx_config.json`, `laya_config.json`, `tokenizer/`. Axes dynamiques batch / séquence / options, opset 17.

`optimize` : shape inference, fusions `ORT_ENABLE_ALL`, INT8 dynamique ou poids FP16.

Sur **CPU Intel / 16 Go**, `int8` est le levier utile. `fp16` est plutôt pour CUDA.

La session CPU active `ORT_ENABLE_ALL`, arena mémoire, mode séquentiel, ≤ 8 threads intra-op. Un chemin `.mlpackage` est rejeté.

## Benchmark PC

```bash
PYTHONPATH=. python benchmarks/pc_benchmark.py ./onnx --calls 2000 --providers cpu
```

Warmup exclu. JSON : `n`, `p50_ms`, `p95_ms`, `mean_ms`.

## Snake

```bash
pip install -e ".[demo]"
laya-onnx-snake --model ./onnx
laya-onnx-snake --model ./onnx --headless --steps 200
laya-onnx-snake --unassisted --record run.jsonl
laya-onnx-snake benchmark --help
laya-onnx-snake export --help
```

Touches : `q` quitter, `espace` pause, `+`/`-` fps, `r` nouvelle partie.

Point d’entrée : `laya_onnx/snake/cli.py` (`play`, `benchmark`, `export`).

## Graphe ONNX

| Tenseur | Shape | Dtype |
| --- | --- | --- |
| `input_ids` | `[B, L]` | int64 ou int32 |
| `attention_mask` | `[B, L]` | idem |
| `marker_pos` | `[B, K]` | idem |
| `marker_mask` | `[B, K]` | bool |
| `qtype` | `[B]` | int (`choice=0`, `score=1`, `noul=2`) |
| `logits` | `[B, K]` | float (slots masqués ≈ -1e4) |
| `act` / `act_probs` | `[B, 2]` | float |

Padding séquence : multiple de 16.

Layout : `[CLS] <type> question: instructions [SEP] [MASK] opt0 … [SEP] state [SEP]`

## Layout

| Fichier | Rôle |
| --- | --- |
| `laya_onnx/agent.py` | `InferenceSession`, providers, threads, `predict` |
| `laya_onnx/convert.py` | torch → ONNX |
| `laya_onnx/optimize.py` | fusions + INT8 / FP16 |
| `laya_onnx/inputs.py` | collate, pad ×16 |
| `laya_onnx/hub.py` | bundle local / HF |
| `laya_onnx/torch_model.py` | `DecisionModel` |
| `laya_onnx/common.py` | prompt, températures |
| `laya_onnx/cli.py` | `predict` / `convert` / `optimize` |
| `laya_onnx/snake/cli.py` | démo live + benchmark + export |
| `tests/test_onnx_export.py` | parité torch ↔ ONNX |
| `benchmarks/pc_benchmark.py` | P50 / P95 |

## Tests

```bash
pip install -e ".[export,dev]"
PYTHONPATH=. pytest -q tests/test_onnx_export.py
```

## Attribution

- Poids : [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) — Apache-2.0
- Bundle ONNX : [receptron/laya-onnx](https://huggingface.co/receptron/laya-onnx)
- Layout / Snake : [laya-mlx](https://github.com/mizorewww/laya-mlx)
- Amont : [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)

Voir LICENSE et NOTICE.

---

## English

ONNX Runtime port of Laya typed System-1 decisions. No text generation. Weights stay on Hugging Face.

```bash
pip install -e ".[demo]"
laya-onnx predict --state-file examples/state.json --questions examples/questions.json
laya-onnx convert --model convaiinnovations/laya --output onnx --precision fp32
laya-onnx optimize ./onnx --precision int8
laya-onnx-snake --model ./onnx
PYTHONPATH=. python benchmarks/pc_benchmark.py ./onnx --calls 200 --providers cpu
```

On Intel CPU prefer **int8** over fp16. Session uses `ORT_ENABLE_ALL`, sequential mode, ≤ 8 intra-op threads. Providers: `cpu` | `openvino` | `cuda`.
