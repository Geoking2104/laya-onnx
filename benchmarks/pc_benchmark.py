"""P50/P95 for one short multilingual decision on this PC.

Usage: python benchmarks/pc_benchmark.py <model_dir> [--calls 2000]
"""

import argparse
import json
import statistics
import time

from laya_onnx import load

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--calls", type=int, default=2000)
parser.add_argument("--providers", default="cpu")
args = parser.parse_args()
agent = load(args.model, providers=args.providers)
questions = {"refund": {"type": "noul", "instructions": "Does the customer request a refund?"}}
agent.predict("Warmup state.", questions)  # compile/warm
samples = []
for _ in range(args.calls):
    t0 = time.perf_counter()
    agent.predict("The customer asks for a refund of a duplicate payment.", questions)
    samples.append((time.perf_counter() - t0) * 1000)
samples.sort()
print(
    json.dumps(
        {
            "n": len(samples),
            "p50_ms": samples[len(samples) // 2],
            "p95_ms": samples[int(len(samples) * 0.95)],
            "mean_ms": statistics.fmean(samples),
        },
        indent=2,
    )
)
