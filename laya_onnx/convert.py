"""Export DecisionModel torch → ONNX (dynamic batch/sequence, opset 17)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .inputs import collate_items
from .torch_model import DecisionModel


def _load_torch_checkpoint(source: Path):
    from safetensors.torch import load_file

    encoder_cfg = json.loads((source / "encoder/config.json").read_text())
    agent_cfg = json.loads((source / "rl_agent_config.json").read_text())
    max_len = int(agent_cfg.get("max_len", 512))
    model = DecisionModel(encoder_cfg, agent_cfg, max_len)
    state = load_file(str(source / "model.safetensors"))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, encoder_cfg, agent_cfg


def convert(
    model_id_or_path,
    output,
    *,
    precision="fp32",
    revision=None,
    subfolder=None,
    token=None,
    opset=17,
    batch_size=2,
    max_options=4,
    max_length=None,
):
    output = Path(output).expanduser()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    import torch

    src = Path(model_id_or_path).expanduser()
    if not src.exists():
        from huggingface_hub import snapshot_download

        src = Path(
            snapshot_download(
                str(model_id_or_path),
                token=token,
                revision=revision,
                allow_patterns=[
                    "model.safetensors",
                    "encoder/*",
                    "tokenizer/*",
                    "rl_agent_config.json",
                ],
            )
        )
        if subfolder:
            src /= subfolder

    model, encoder_cfg, agent_cfg = _load_torch_checkpoint(src)
    max_len = int(agent_cfg.get("max_len", max_length or 64))
    dummy_len = min(32, max_len)
    items = [
        {
            "ids": list(range(dummy_len)),
            "markers": list(range(3, 3 + min(2, max_options))),
            "qtype": 0,
        }
        for _ in range(batch_size)
    ]
    shape = {
        "batch_size": batch_size,
        "max_length": max_len,
        "min_length": 16,
        "max_options": max_options,
    }
    arrays = collate_items(items, 0, shape=shape)
    example = {k: torch.from_numpy(v) for k, v in arrays.items()}
    example["marker_mask"] = example["marker_mask"].bool()

    output.mkdir(parents=True)
    try:
        graph = output / "laya.onnx"
        dynamic_axes = {
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "marker_pos": {0: "batch", 1: "options"},
            "marker_mask": {0: "batch", 1: "options"},
            "qtype": {0: "batch"},
            "logits": {0: "batch", 1: "options"},
            "act": {0: "batch"},
        }
        torch.onnx.export(
            model,
            (
                example["input_ids"],
                example["attention_mask"],
                example["marker_pos"],
                example["marker_mask"],
                example["qtype"],
            ),
            str(graph),
            input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
            output_names=["logits", "act"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            dynamo=False,
        )
        from .optimize import optimize_bundle

        optimize_bundle(graph, precision=precision, fuse=True)

        if (src / "tokenizer").is_dir():
            shutil.copytree(src / "tokenizer", output / "tokenizer")
        keys = ("max_len", "head_max_len", "temperature", "temperature_by_options")
        config = {k: agent_cfg[k] for k in keys if k in agent_cfg}
        config.setdefault("max_len", max_len)
        config.setdefault("head_max_len", int(agent_cfg.get("head_max_len", 32)))
        config.setdefault("temperature", [1.0, 1.0, 1.0])
        config.setdefault("temperature_by_options", {})
        (output / "laya_config.json").write_text(json.dumps(config, indent=2) + "\n")
        (output / "onnx_config.json").write_text(
            json.dumps({"format": "laya-onnx", "format_version": 1, "precision": precision, "opset": opset, **config, "encoder": encoder_cfg}, indent=2)
            + "\n"
        )
        (output / "rl_agent_config.json").write_text(json.dumps(agent_cfg, indent=2) + "\n")
        if (src / "encoder").is_dir():
            shutil.copytree(src / "encoder", output / "encoder")
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return output
