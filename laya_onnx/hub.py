"""Resolve a local ONNX bundle or a Hugging Face snapshot."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from huggingface_hub import snapshot_download

GRAPH_NAMES = ("laya.onnx", "model.onnx")
CONFIG_NAMES = ("onnx_config.json", "laya_config.json", "rl_agent_config.json")


def bundle_dir(model_id: str, revision=None) -> Path:
    """A real, non-symlinked directory for a downloaded bundle.

    ONNX Runtime validates that a model's external-data file stays inside the
    model directory. The default Hugging Face cache stores snapshot entries as
    symlinks into ``blobs/``, so that validation fails (seen on Windows);
    downloading into a ``local_dir`` materialises regular files instead.
    """
    from huggingface_hub.constants import HF_HUB_CACHE

    slug = str(model_id).replace("/", "--")
    if revision:
        slug += "@" + str(revision).replace("/", "--")
    return Path(HF_HUB_CACHE) / "laya-onnx-bundles" / slug


def find_graph(path: Path) -> Path:
    for name in GRAPH_NAMES:
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No ONNX graph (laya.onnx / model.onnx) under {path}")


def find_config(path: Path) -> Path:
    for name in CONFIG_NAMES:
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No onnx_config.json / laya_config.json under {path}")


def resolve_bundle(model_id_or_path, *, token=None, subfolder=None, revision=None) -> Path:
    if subfolder:
        part = PurePosixPath(subfolder)
        if part.is_absolute() or ".." in part.parts:
            raise ValueError("subfolder must be a relative path inside the model repository")
    path = Path(model_id_or_path).expanduser()
    if path.exists():
        if path.suffix == ".mlpackage" or str(path).endswith(".mlpackage"):
            raise ValueError(
                f"{path} looks like a Core ML package (.mlpackage). "
                "laya-onnx expects an ONNX bundle (laya.onnx + tokenizer/). "
                "Convert with `laya-onnx convert` or point to receptron/laya-onnx."
            )
        if subfolder:
            path /= subfolder
        find_graph(path)
        return path
    value = str(model_id_or_path)
    if value.startswith(("/", "./", "../", "~")) or isinstance(model_id_or_path, Path):
        raise FileNotFoundError(f"Local model directory does not exist: {value}")
    prefix = subfolder.rstrip("/") + "/" if subfolder else ""
    patterns = [
        prefix + name
        for name in (
            "laya.onnx",
            "laya.onnx.data",
            "model.onnx",
            "model.onnx.data",
            "laya_config.json",
            "onnx_config.json",
            "rl_agent_config.json",
            "tokenizer/*",
        )
    ]
    path = Path(snapshot_download(value, token=token, revision=revision, allow_patterns=patterns, local_dir=bundle_dir(value, revision)))
    if subfolder:
        path /= subfolder
    find_graph(path)
    return path
