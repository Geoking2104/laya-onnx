import json

from laya_onnx.verify import load_manifest, sha256_file, verify_bundle


def _manifest(tmp_path, payload: bytes):
    file_path = tmp_path / "laya_config.json"
    file_path.write_bytes(payload)
    manifest = {
        "algorithm": "sha256",
        "files": {"laya_config.json": {"size": len(payload), "sha256": sha256_file(file_path)}},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return file_path, manifest_path


def test_verify_bundle_ok(tmp_path):
    _, manifest_path = _manifest(tmp_path, b'{"max_len": 64}\n')
    ok, rows = verify_bundle(tmp_path, manifest_path)
    assert ok is True
    assert rows == [{"file": "laya_config.json", "status": "ok", "sha256": rows[0]["sha256"]}]


def test_verify_bundle_detects_tampering(tmp_path):
    file_path, manifest_path = _manifest(tmp_path, b'{"max_len": 64}\n')
    file_path.write_bytes(b'{"max_len": 128}\n')  # same directory, changed bytes
    ok, rows = verify_bundle(tmp_path, manifest_path)
    assert ok is False
    assert rows[0]["status"] == "size-mismatch"


def test_verify_bundle_hash_mismatch(tmp_path):
    file_path, manifest_path = _manifest(tmp_path, b"abc")
    file_path.write_bytes(b"abd")  # same size, different bytes
    ok, rows = verify_bundle(tmp_path, manifest_path)
    assert ok is False
    assert rows[0]["status"] == "sha256-mismatch"


def test_verify_bundle_missing_file(tmp_path):
    _, manifest_path = _manifest(tmp_path, b"x")
    (tmp_path / "laya_config.json").unlink()
    ok, rows = verify_bundle(tmp_path, manifest_path)
    assert ok is False
    assert rows[0]["status"] == "missing"


def test_bundled_manifest_is_wellformed():
    manifest = load_manifest()
    assert manifest["repo"] == "receptron/laya-onnx"
    assert manifest["algorithm"] == "sha256"
    assert "laya.onnx" in manifest["files"]
    for name, entry in manifest["files"].items():
        assert len(entry["sha256"]) == 64 and entry["size"] > 0
