import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from laya_onnx.inputs import collate_items
from laya_onnx.torch_model import DecisionModel


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(2026)
    cfg = {
        "model_type": "modernbert",
        "hidden_size": 32,
        "intermediate_size": 48,
        "vocab_size": 100,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "local_attention": 8,
        "global_attn_every_n_layers": 3,
    }
    agent = {
        "head_layers": 2,
        "act_costs": {"escalate": 0.5},
        "max_len": 64,
        "head_max_len": 32,
        "encoder": "tiny-test",
        "temperature": [1.0, 1.0, 1.0],
    }
    model = DecisionModel(cfg, agent, 64).eval()
    for name, p in model.named_parameters():
        if "in_proj" in name:
            torch.nn.init.normal_(p, std=0.03)
    (tmp_path / "encoder").mkdir()
    (tmp_path / "encoder/config.json").write_text(json.dumps(cfg))
    (tmp_path / "rl_agent_config.json").write_text(json.dumps(agent))
    (tmp_path / "tokenizer").mkdir()
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    tok = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[PAD]": 4},
            unk_token="[UNK]",
        )
    )
    tok.save(str(tmp_path / "tokenizer/tokenizer.json"))
    (tmp_path / "tokenizer/tokenizer_config.json").write_text(
        json.dumps(
            {
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "mask_token": "[MASK]",
                "pad_token": "[PAD]",
            }
        )
    )
    save_file(model.state_dict(), str(tmp_path / "model.safetensors"))
    return tmp_path, model


def batch(length=31, batch_size=3):
    items = [
        {"ids": list(range(length)), "markers": [3, 8, 10][: i + 1], "qtype": i}
        for i in range(batch_size)
    ]
    shape = {"batch_size": 3, "max_length": 64, "min_length": 16, "max_options": 4}
    return {k: torch.from_numpy(v) for k, v in collate_items(items, 4, shape=shape).items()}


def test_onnx_export_parity_and_dynamic_lengths(checkpoint):
    from laya_onnx import Agent
    from laya_onnx.convert import convert

    source, model = checkpoint
    output = source / "export"
    convert(source, output, batch_size=3, max_options=4)
    agent = Agent(output)
    for length in (17, 47, 64):
        inputs = batch(length)
        with torch.inference_mode():
            expected = model(**inputs)
        actual = agent.forward({k: v.numpy() for k, v in inputs.items()})
        for a, e in zip(actual, expected):
            np.testing.assert_allclose(a, e.numpy(), atol=1e-4, rtol=1e-4)
    result = agent.predict(
        "sample",
        {"q": {"type": "choice", "instructions": "pick", "criteria": ["a", "b"]}},
    )
    assert result["usage"]["output_tokens"] == 0
    assert set(result["answers"]["q"]["probabilities"]) == {"a", "b"}
    with pytest.raises(FileExistsError):
        convert(source, output)
