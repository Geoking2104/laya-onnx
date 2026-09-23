"""Verify a downloaded ONNX bundle against a SHA-256 manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MANIFEST_DIR = Path(__file__).with_name("checksums")
DEFAULT_MANIFEST = MANIFEST_DIR / "receptron-laya-onnx.json"

CHUNK = 1 << 20


def sha256_file(path, chunk=CHUNK) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path=None) -> dict:
    manifest = Path(path) if path else DEFAULT_MANIFEST
    if not manifest.is_file():
        raise FileNotFoundError(f"No checksum manifest at {manifest}")
    return json.loads(manifest.read_text(encoding="utf-8"))


def verify_bundle(path, manifest=None):
    """Check every file in the manifest against ``path``.

    Returns ``(ok, rows)`` where each row is ``{"file", "status", ...}`` and
    status is one of ``ok`` / ``missing`` / ``size-mismatch`` / ``sha256-mismatch``.
    """
    data = manifest if isinstance(manifest, dict) else load_manifest(manifest)
    root = Path(path).expanduser()
    rows = []
    ok = True
    for name, expected in data.get("files", {}).items():
        target = root / name
        if not target.is_file():
            rows.append({"file": name, "status": "missing"})
            ok = False
            continue
        size = target.stat().st_size
        if expected.get("size") is not None and size != expected["size"]:
            rows.append(
                {
                    "file": name,
                    "status": "size-mismatch",
                    "expected_size": expected["size"],
                    "size": size,
                }
            )
            ok = False
            continue
        digest = sha256_file(target)
        if expected.get("sha256") and digest != expected["sha256"]:
            rows.append(
                {
                    "file": name,
                    "status": "sha256-mismatch",
                    "expected": expected["sha256"],
                    "actual": digest,
                }
            )
            ok = False
        else:
            rows.append({"file": name, "status": "ok", "sha256": digest})
    return ok, rows
