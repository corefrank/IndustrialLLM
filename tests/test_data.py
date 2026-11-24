import hashlib
import json
from pathlib import Path

import pytest

from industrialllm import data
from industrialllm.schema import FIELDS, Extraction, read_jsonl


def raw(text):
    words = text.split()
    return {"text": text, "tokens": words,
            "entities": [{"start": 0, "end": 1, "type": label} for label in data.LABEL_FIELDS],
            "relations": []}


def test_raw_token_span_and_five_keys():
    row = raw("replace valve")
    row["entities"] = [
        {"type": "Activity/MaintenanceActivity/Replace", "start": 0, "end": 1},
        {"type": "PhysicalObject/DrivingObject", "start": 1, "end": 2},
    ]
    result = data.convert(row, "gold:1")
    assert tuple(result["target"]) == FIELDS
    assert result["target"]["physical_objects"] == ["valve"]
    assert result["target"]["activities"] == ["replace"]
    assert result["target"]["states"] == []
    assert Extraction.model_validate_json(result["completion"][0]["content"]).model_dump() == result["target"]


@pytest.mark.parametrize("start,end", [(-1, 1), (1, 1), (0, 5)])
def test_reject_invalid_span(start, end):
    row = raw("replace valve")
    row["entities"][0].update(start=start, end=end)
    with pytest.raises(ValueError, match="invalid token span"):
        data.convert(row, "gold:1")


def test_reject_token_mismatch():
    row = raw("TBD")
    row["tokens"] = ["repair"]
    with pytest.raises(ValueError, match="grounded"):
        data.convert(row, "gold:1")


def test_all_splits_and_repeatable_files(tmp_path, monkeypatch):
    seed_dir = tmp_path / "raw"
    seed_dir.mkdir()
    gold = [raw(text) for text in ("inspect the primary motor mount", "inspect the secondary motor mount",
                                  "tighten fastener", "change oil", "inspect line", "check drum")]
    silver = [raw(text) for text in ("inspect the broken motor mount", "change oil", "Change   oil",
                                    "replace hydraulic hose", "new filter", "swap bearing")]
    for name, values in (("gold", gold), ("silver", silver)):
        payload = json.dumps(values).encode()
        (seed_dir / f"{name}_release.json").write_bytes(payload)
        monkeypatch.setitem(data.FILES, name, hashlib.sha256(payload).hexdigest())
    settings = {"train": 2, "validation": 2, "test": 2, "seed": 42}
    a, b = tmp_path / "a", tmp_path / "b"
    data.prepare(seed_dir, a, settings)
    data.prepare(seed_dir, b, settings)
    for path in a.iterdir():
        assert path.read_bytes() == (b / path.name).read_bytes()
    exclusions = {r["reason"] for r in read_jsonl(a / "excluded.jsonl")}
    assert {"gold_near_overlap", "gold_exact_overlap", "exact_normalized_duplicate"} <= exclusions


def test_official_artifacts_if_prepared():
    path = Path("data/processed")
    if not path.exists():
        pytest.skip("Official corpus is exercised separately when freshly downloaded.")
    manifest = json.loads((path / "split_manifest.json").read_text())
    assert [len(manifest["splits"][k]) for k in ("train", "validation", "test")] == [2400, 250, 400]
    for split in ("train", "validation", "test"):
        rows = read_jsonl(path / f"{split}.jsonl")
        assert all(Extraction.model_validate(row["target"]) for row in rows)
