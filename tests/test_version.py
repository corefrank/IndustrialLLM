import pytest

from industrialllm.data import sha
from industrialllm.model import load_manifest
from industrialllm.schema import write_json


def test_version_integrity(tmp_path):
    directory = tmp_path / "adapter"
    directory.mkdir()
    artifact = directory / "adapter_model.safetensors"
    artifact.write_bytes(b"dummy adapter for checksum test")
    config = {"model": {"id": "qwen", "revision": "fixed"}}
    path = tmp_path / "manifest.json"
    write_json(path, {
        "base_model": "qwen", "model_revision": "fixed", "adapter": "adapter",
        "adapter_sha256": {artifact.name: sha(artifact)},
    })
    assert load_manifest(path, config)[0] == directory
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_manifest(path, config)
