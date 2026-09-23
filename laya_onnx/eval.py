"""Decision evaluation and calibration metrics.

Runs a labeled eval set through a loaded :class:`laya_onnx.Agent` and reports
per-question-type accuracy, score MAE, confidence calibration (ECE) and latency
percentiles. Everything here is pure/offline so it can be unit-tested with a
stub agent.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

DEFAULT_BINS = 10


def percentile(sorted_values, fraction):
    """Nearest-rank percentile. ``sorted_values`` must already be sorted."""
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1, int(len(sorted_values) * fraction))
    return sorted_values[index]


def bin_index(value, bins=DEFAULT_BINS):
    return min(bins - 1, max(0, int(value * bins)))


def expected_calibration_error(confidences, correct, bins=DEFAULT_BINS):
    """Expected Calibration Error over (confidence, correctness) pairs.

    Lower is better; 0 means confidence matches accuracy exactly.
    """
    if not confidences:
        return None
    total = len(confidences)
    error = 0.0
    for b in range(bins):
        low = b / bins
        high = (b + 1) / bins
        members = [
            i
            for i, confidence in enumerate(confidences)
            if low <= confidence < high or (b == bins - 1 and confidence >= 1.0)
        ]
        if not members:
            continue
        accuracy = sum(1.0 for i in members if correct[i]) / len(members)
        mean_confidence = sum(confidences[i] for i in members) / len(members)
        error += (len(members) / total) * abs(accuracy - mean_confidence)
    return error


def score_answer(qtype, answer, gold):
    """Return ``(correct, error)`` for one answered question.

    ``correct`` is ``None`` when the item has no scoreable answer, ``error`` is
    only set for ``score`` questions.
    """
    if answer is None or gold is None:
        return None, None
    if qtype == "choice":
        return (answer.get("choice") == gold), None
    if qtype == "noul":
        try:
            predicted = float(answer.get("noul", 0.0)) >= 0.5
        except (TypeError, ValueError):
            return None, None
        return (bool(predicted) == bool(gold)), None
    if qtype == "score":
        try:
            error = abs(float(answer.get("score", 0.0)) - float(gold))
        except (TypeError, ValueError):
            return None, None
        return (error <= 0.5), error
    return None, None


def load_items(path):
    """Read a JSONL eval set (one ``{"state", "questions", "gold"}`` per line)."""
    items = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: {exc}")
    return items


def run_eval(agent, items, *, bins=DEFAULT_BINS):
    """Run ``items`` through ``agent`` and return a results dict."""
    latencies = []
    records = []
    for item in items:
        started = time.perf_counter()
        out = agent.predict(item["state"], item["questions"])
        latencies.append((time.perf_counter() - started) * 1000.0)
        answers = out.get("answers", {})
        gold = item.get("gold") or {}
        for qid, definition in item["questions"].items():
            answer = answers.get(qid, {})
            qtype = definition["type"]
            correct, error = score_answer(qtype, answer, gold.get(qid))
            records.append(
                {
                    "type": qtype,
                    "correct": correct,
                    "error": error,
                    "confidence": float(answer.get("confidence", 0.0) or 0.0),
                }
            )

    buckets = {}
    for record in records:
        bucket = buckets.setdefault(
            record["type"], {"confidences": [], "outcomes": [], "errors": [], "n": 0}
        )
        bucket["n"] += 1
        if record["correct"] is not None:
            bucket["confidences"].append(record["confidence"])
            bucket["outcomes"].append(bool(record["correct"]))
        if record["error"] is not None:
            bucket["errors"].append(record["error"])

    by_type = {}
    for qtype, bucket in buckets.items():
        entry = {"n": bucket["n"]}
        if bucket["outcomes"]:
            entry["accuracy"] = round(sum(bucket["outcomes"]) / len(bucket["outcomes"]), 4)
            entry["ece"] = round(
                expected_calibration_error(bucket["confidences"], bucket["outcomes"], bins), 4
            )
        if bucket["errors"]:
            entry["mae"] = round(statistics.fmean(bucket["errors"]), 4)
        by_type[qtype] = entry

    confidences = [r["confidence"] for r in records if r["correct"] is not None]
    outcomes = [bool(r["correct"]) for r in records if r["correct"] is not None]
    latencies.sort()
    overall = {
        "items": len(items),
        "questions": len(records),
        "answers_scored": len(outcomes),
        "accuracy": round(sum(outcomes) / len(outcomes), 4) if outcomes else None,
        "ece": round(expected_calibration_error(confidences, outcomes, bins), 4)
        if confidences
        else None,
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 3) if latencies else None,
            "p50": round(percentile(latencies, 0.5), 3) if latencies else None,
            "p95": round(percentile(latencies, 0.95), 3) if latencies else None,
        },
    }
    return {"by_type": by_type, "overall": overall}


def format_markdown(results, title="laya-onnx decision eval"):
    """Render a results dict (from :func:`run_eval`) as Markdown."""
    overall = results.get("overall", {})
    latency = overall.get("latency_ms", {})
    lines = [
        f"### {title}",
        "",
        f"- items: **{overall.get('items')}**  questions: **{overall.get('questions')}**",
        f"- accuracy: **{overall.get('accuracy')}**  ECE: **{overall.get('ece')}**",
        f"- latency (per call, ms): mean {latency.get('mean')} / p50 {latency.get('p50')} / p95 {latency.get('p95')}",
        "",
        "| question type | n | accuracy | MAE | ECE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for qtype, entry in sorted(results.get("by_type", {}).items()):
        lines.append(
            "| {0} | {1} | {2} | {3} | {4} |".format(
                qtype,
                entry.get("n", ""),
                entry.get("accuracy", "-"),
                entry.get("mae", "-"),
                entry.get("ece", "-"),
            )
        )
    return "\n".join(lines)
