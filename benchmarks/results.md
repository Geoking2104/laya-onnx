# Benchmarks

Reproducible decision benchmarks for `laya-onnx`. Everything here runs offline
once weights are cached; nothing is a leaderboard — these are self-checks you
can rerun on your own machine.

## Decision eval (accuracy + calibration)

```bash
# local bundle
python benchmarks/evaluate.py ./onnx --json benchmarks/results.json
# or an already-cached Hub bundle
python benchmarks/evaluate.py receptron/laya-onnx
```

The eval set (`benchmarks/eval/decisions.jsonl`) is a small labeled self-check:
**8 states / 10 questions** across the three typed forms (`choice`, `score`,
`noul`). Confidence values are the model's own; calibration is reported as
expected calibration error (ECE, 10 bins, lower is better).

### Published result

Environment: Windows 11, 12th Gen Intel Core i7-1255U (12 threads, 15.8 GB RAM),
`onnxruntime` CPU, bundle `receptron/laya-onnx` @ `68f27dfe`.

| question type | n | accuracy | MAE | ECE |
| --- | ---: | ---: | ---: | ---: |
| choice | 4 | 1.00 | – | 0.456 |
| noul | 4 | 1.00 | – | 0.129 |
| score | 2 | 0.50 | 0.530 | 0.112 |
| **overall** | **10** | **0.90** | – | **0.256** |

Latency per `predict()` call: mean **2808 ms** / p50 **1760 ms** / p95 **7418 ms**.

Reading:
- All four `choice` and all four `noul` decisions were correct; one of the two
  `score` items missed (MAE 0.53).
- Calibration is **not** tight at this sample size. The `choice` bucket is
  *under*-confident (correct, but ~0.55 confidence), which is recorded rather
  than hidden. The checkpoint also ships one calibration temperature outside the
  accepted range; `load()` warns and clamps it (expected, not an error).
- With 10 questions the confidence intervals are wide — treat this as a smoke
  result, not a measurement of model quality.

Raw output: [`benchmarks/results.json`](results.json).

## Latency

```bash
python benchmarks/pc_benchmark.py ./onnx --calls 2000
```

P50/P95 for one short multilingual decision on this machine (no eval set
needed). See the note above for a measured sample.

## Verify a download

Every published bundle ships a SHA-256 manifest under `laya_onnx/checksums/`.
Check a local copy:

```bash
laya-onnx verify --model ~/.cache/huggingface/hub/laya-onnx-bundles/receptron--laya-onnx
```

`receptron/laya-onnx` @ `68f27dfe` — sizes and SHA-256:

| file | bytes | sha256 |
| --- | ---: | --- |
| `laya.onnx` | 3,807,291 | `a874eb25…66dba1e` |
| `laya.onnx.data` | 1,685,258,240 | `48774636…3242aba` |
| `laya_config.json` | 369 | `5049005d…69bb561` |
| `tokenizer/tokenizer.json` | 3,583,228 | `6c8aaa9a…3c08d30` |
| `tokenizer/tokenizer_config.json` | 308 | `50044de6…320dd12` |

The two weight files match the SHA-256 published by Hugging Face for their LFS
objects; full digests are in
[`laya_onnx/checksums/receptron-laya-onnx.json`](../laya_onnx/checksums/receptron-laya-onnx.json).
