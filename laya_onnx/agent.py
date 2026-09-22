"""ONNX Runtime inference; prompt and result formats follow upstream Laya."""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np

from .common import (
    QTYPES,
    TEMP_MAX,
    TEMP_MIN,
    build_sequence,
    clamp_temperature,
    confidence_from_probs,
    render_options,
    temp_bucket,
)
from .hub import find_config, find_graph, resolve_bundle
from .inputs import collate_items
from .tokenizer import Tokenizer


def _providers(name: str) -> list[str]:
    if name == "openvino":
        return ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    if name == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _session_dtypes(session) -> dict[str, np.dtype]:
    mapping = {}
    for inp in session.get_inputs():
        raw = (inp.type or "").lower()
        if "int32" in raw:
            mapping[inp.name] = np.int32
        elif "int64" in raw:
            mapping[inp.name] = np.int64
        elif "bool" in raw:
            mapping[inp.name] = np.bool_
        elif "float16" in raw:
            mapping[inp.name] = np.float16
        else:
            mapping[inp.name] = None
    return mapping


class Agent:
    def __init__(
        self,
        model_id_or_path="receptron/laya-onnx",
        providers="cpu",
        token=None,
        subfolder=None,
        *,
        revision=None,
        batch_size=16,
        threads=None,
        pad_to_multiple=16,
        deterministic=False,
    ):
        if deterministic:
            threads = 1
            pad_to_multiple = None
        if providers not in ("cpu", "openvino", "cuda"):
            raise ValueError("providers must be 'cpu', 'openvino', or 'cuda'")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if pad_to_multiple is not None and (
            not isinstance(pad_to_multiple, int)
            or isinstance(pad_to_multiple, bool)
            or pad_to_multiple < 1
        ):
            raise ValueError("pad_to_multiple must be a positive integer or None")
        self.providers = providers
        self.batch_size = batch_size
        self.deterministic = bool(deterministic)
        self.pad_to_multiple = pad_to_multiple
        self.model_id = str(model_id_or_path)
        self.revision = revision
        self.model_dir = resolve_bundle(
            model_id_or_path, token=token, subfolder=subfolder, revision=revision
        )
        cfg_path = find_config(self.model_dir)
        raw = json.loads(cfg_path.read_text())
        self.cfg = raw
        max_len = int(raw.get("max_len", 512))
        head_max_len = int(raw.get("head_max_len", 192))
        if not 4 < head_max_len < max_len:
            raise ValueError("Expected 4 < head_max_len < max_len")
        self.temperature_raw = raw.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options_raw = raw.get("temperature_by_options", {})
        if len(self.temperature_raw) != 3 or any(
            not math.isfinite(float(t)) or float(t) <= 0
            for t in [*self.temperature_raw, *self.temperature_by_options_raw.values()]
        ):
            raise ValueError("Calibration temperatures must be finite and positive")
        self.temperature = [clamp_temperature(t) for t in self.temperature_raw]
        self.temperature_by_options = {
            k: clamp_temperature(v) for k, v in self.temperature_by_options_raw.items()
        }
        rejected = [
            "%s=%.4g" % (k, float(v))
            for k, v in self.temperature_by_options_raw.items()
            if clamp_temperature(v) != float(v)
        ]
        rejected += [
            "temperature[%d]=%.4g" % (i, float(t))
            for i, t in enumerate(self.temperature_raw)
            if clamp_temperature(t) != float(t)
        ]
        if rejected:
            warnings.warn(
                "laya-onnx: this checkpoint ships temperatures outside [%g, %g] which would "
                "distort confidence; clamping %s." % (TEMP_MIN, TEMP_MAX, ", ".join(rejected)),
                RuntimeWarning,
                stacklevel=2,
            )
        self.tok = Tokenizer(self.model_dir / "tokenizer")

        import os

        import onnxruntime as ort

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.enable_mem_pattern = True
        sess_opts.enable_cpu_mem_arena = not self.deterministic
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        workers = int(threads) if threads else min(8, os.cpu_count() or 4)
        sess_opts.intra_op_num_threads = workers
        sess_opts.inter_op_num_threads = 1
        sess_opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            str(find_graph(self.model_dir)),
            sess_options=sess_opts,
            providers=_providers(providers),
        )
        self._dtypes = _session_dtypes(self.session)

    @staticmethod
    def _to_internal(qdef):
        if not isinstance(qdef, dict):
            raise ValueError("Each question must be a dictionary")
        kind = qdef.get("type")
        if kind not in QTYPES:
            raise ValueError(f"Unknown question type {kind!r}; expected choice, score, or noul")
        if "instructions" not in qdef:
            raise ValueError("Question is missing instructions")
        criteria = qdef.get("criteria")
        if kind == "choice":
            if isinstance(criteria, list):
                if not all(isinstance(c, str) for c in criteria):
                    raise ValueError("Choice labels must be strings")
                if len(set(criteria)) != len(criteria):
                    raise ValueError("Choice labels must be unique")
                criteria = dict.fromkeys(criteria)
            if not isinstance(criteria, dict) or not criteria:
                raise ValueError("Choice criteria must be a nonempty dictionary or list")
            if not all(isinstance(k, str) for k in criteria):
                raise ValueError("Choice labels must be strings")
        elif kind == "score":
            if not isinstance(criteria, list) or not criteria:
                raise ValueError("Score criteria must be a nonempty list")
        elif criteria is not None and not isinstance(criteria, dict):
            raise ValueError("Noul criteria must be a dictionary with false/true descriptions")
        instructions = qdef["instructions"]
        if not isinstance(instructions, str):
            instructions = json.dumps(instructions)
        return {"t": kind, "ins": instructions, "crit": criteria}

    def prepare(self, state, questions):
        if not isinstance(questions, dict):
            raise ValueError("questions must be a dictionary keyed by question id")
        items, internal = [], []
        for qid, definition in questions.items():
            q = self._to_internal(definition)
            ids, markers = build_sequence(
                self.tok, state, q, self.cfg.get("max_len", 512), self.cfg.get("head_max_len", 192)
            )
            if len(markers) != len(render_options(q)):
                raise ValueError(f"Question {qid!r} has too many options for the token budget")
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
            internal.append(q)
        return items, internal

    def _cast(self, batch: dict) -> dict:
        feeds = {}
        for name, array in batch.items():
            target = self._dtypes.get(name)
            if target is not None and array.dtype != target:
                array = array.astype(target, copy=False)
            feeds[name] = array
        return feeds

    def forward(self, batch):
        feeds = self._cast(batch)
        names = [o.name for o in self.session.get_outputs()]
        outputs = self.session.run(names, feeds)
        return tuple(np.asarray(o) for o in outputs)

    def system_one(self, state, questions, *, argmax=False):
        items, internal = self.prepare(state, questions)
        answers = {}
        question_ids = list(questions)
        id_dt = self._dtypes.get("input_ids") or np.int64
        mk_dt = self._dtypes.get("marker_pos") or np.int64
        for start in range(0, len(items), self.batch_size):
            chunk = items[start : start + self.batch_size]
            batch = collate_items(
                chunk,
                self.tok.pad_token_id,
                pad_to_multiple=self.pad_to_multiple,
                max_length=self.cfg.get("max_len", 512),
                ids_dtype=id_dt,
                marker_dtype=mk_dt,
            )
            logits, act = self.forward(batch)
            if not np.isfinite(logits).all() or not np.isfinite(act).all():
                raise FloatingPointError("Non-finite model outputs")
            if act.ndim == 2 and act.shape[-1] > 1 and abs(float(act[0].sum()) - 1.0) > 1e-3:
                act = np.exp(act - act.max(axis=-1, keepdims=True))
                act /= act.sum(axis=-1, keepdims=True)
            for row, item in enumerate(chunk):
                qid, q = question_ids[start + row], internal[start + row]
                k, qt = len(item["markers"]), item["qtype"]
                scale = (
                    1.0
                    if argmax
                    else self.temperature_by_options.get(temp_bucket(qt, k), self.temperature[qt])
                )
                z = logits[row, :k] / scale
                p = np.exp(z - z.max())
                p /= p.sum()
                act_p = float(act[row, 0]) if act.ndim == 2 else float(act[row])
                answer = {
                    "type": q["t"],
                    "confidence": round(confidence_from_probs(p, k), 4),
                    "action": {"act_probability": round(act_p, 4)},
                }
                if q["t"] == "choice":
                    labels = list(q["crit"])
                    answer.update(
                        choice=labels[int(p.argmax())],
                        probabilities={label: round(float(v), 4) for label, v in zip(labels, p)},
                    )
                elif q["t"] == "score":
                    answer.update(
                        score=round(float((np.arange(k) * p).sum()), 4),
                        legend={str(i): value for i, value in enumerate(q["crit"])},
                        probabilities={str(i): round(float(v), 4) for i, v in enumerate(p)},
                    )
                else:
                    answer.update(
                        noul=round(float(p[1]), 4),
                        confidence=round(max(float(p[1]), 1.0 - float(p[1])), 4),
                    )
                answers[qid] = answer
        return {
            "model": "laya-onnx",
            "answers": answers,
            "usage": {"input_tokens": sum(len(item["ids"]) for item in items), "output_tokens": 0},
        }

    predict = system_one

    def predict_argmax(self, state, questions):
        """Deterministic path: ignore calibrated temperature (see docs/DETERMINISTIC.md)."""
        return self.system_one(state, questions, argmax=True)


RLAgent = Agent


def load(model_id_or_path="receptron/laya-onnx", providers="cpu", token=None, subfolder=None, deterministic=False, **kwargs):
    return Agent(model_id_or_path, providers=providers, token=token, subfolder=subfolder, deterministic=deterministic, **kwargs)
