"""Dynamic batch collation. Pad sequence length to a multiple of 16."""

from __future__ import annotations

import numpy as np


def pad_length(length: int, multiple: int = 16, max_length: int | None = None) -> int:
    padded = ((length + multiple - 1) // multiple) * multiple
    if max_length is not None:
        padded = min(max(padded, multiple), max_length)
    return max(padded, multiple if length else 1)


def collate_items(
    items,
    pad_id,
    *,
    shape=None,
    pad_to_multiple=16,
    max_length=None,
    marker_dtype=None,
    ids_dtype=None,
):
    if not items:
        raise ValueError("Cannot collate an empty batch")
    if shape:
        batch_size = int(shape.get("batch_size", len(items)))
        max_length = int(shape.get("max_length", max_length or 512))
        min_length = int(shape.get("min_length", pad_to_multiple))
        max_options = int(shape.get("max_options", max(2, max(len(i["markers"]) for i in items))))
        pad_to_multiple = min_length if min_length else pad_to_multiple
    else:
        batch_size = len(items)
        max_options = max(2, max(len(i["markers"]) for i in items))

    length = max(len(item["ids"]) for item in items)
    if pad_to_multiple:
        length = pad_length(length, pad_to_multiple, max_length)
    if shape:
        length = max(length, int(shape.get("min_length", length)))
        if max_length is not None:
            length = min(length, max_length)

    id_dt = ids_dtype or np.int64
    mk_dt = marker_dtype or np.int64
    batch = {
        "input_ids": np.full((batch_size, length), pad_id, dtype=id_dt),
        "attention_mask": np.zeros((batch_size, length), dtype=id_dt),
        "marker_pos": np.zeros((batch_size, max_options), dtype=mk_dt),
        "marker_mask": np.zeros((batch_size, max_options), dtype=np.bool_),
        "qtype": np.zeros((batch_size,), dtype=id_dt),
    }
    for i, item in enumerate(items[:batch_size]):
        seq, markers = item["ids"], item["markers"]
        seq = seq[:length]
        batch["input_ids"][i, : len(seq)] = seq
        batch["attention_mask"][i, : len(seq)] = 1
        count = min(len(markers), max_options)
        batch["marker_pos"][i, :count] = markers[:count]
        batch["marker_mask"][i, :count] = True
        batch["qtype"][i] = item["qtype"]
    return batch
