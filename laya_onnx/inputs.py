"""Dynamic batch collation. Pad sequence length to a multiple of 16."""

from __future__ import annotations

import numpy as np


def pad_length(length: int, multiple: int = 16, max_length: int | None = None) -> int:
    padded = ((length + multiple - 1) // multiple) * multiple
    if max_length is not None:
        padded = min(padded, max_length)
    return max(padded, multiple if length else 1)


def collate_items(items, pad_id, *, pad_to_multiple=16, max_length=None, marker_dtype=None, ids_dtype=None):
    if not items:
        raise ValueError("Cannot collate an empty batch")
    n = len(items)
    length = max(len(item["ids"]) for item in items)
    if pad_to_multiple:
        length = pad_length(length, pad_to_multiple, max_length)
    count = max(2, max(len(item["markers"]) for item in items))
    id_dt = ids_dtype or np.int64
    mk_dt = marker_dtype or np.int64
    batch = {
        "input_ids": np.full((n, length), pad_id, dtype=id_dt),
        "attention_mask": np.zeros((n, length), dtype=id_dt),
        "marker_pos": np.zeros((n, count), dtype=mk_dt),
        "marker_mask": np.zeros((n, count), dtype=np.bool_),
        "qtype": np.array([item["qtype"] for item in items], dtype=id_dt),
    }
    for i, item in enumerate(items):
        seq, markers = item["ids"], item["markers"]
        batch["input_ids"][i, : len(seq)] = seq
        batch["attention_mask"][i, : len(seq)] = 1
        batch["marker_pos"][i, : len(markers)] = markers
        batch["marker_mask"][i, : len(markers)] = True
    return batch
