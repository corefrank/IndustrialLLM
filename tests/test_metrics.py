import json

import pytest
from pydantic import ValidationError

from industrialllm.evaluate import score, summarize
from industrialllm.schema import FIELDS, Extraction, grounded, strict_json


def target(**kwargs):
    return dict.fromkeys(FIELDS, []) | kwargs


def test_schema_and_json_are_strict():
    with pytest.raises(ValidationError):
        Extraction.model_validate(target(activities="replace"))
    with pytest.raises(ValidationError):
        Extraction.model_validate(target(priority=[]))
    with pytest.raises(ValueError):
        strict_json('{"a":1,"a":2}')
    with pytest.raises(ValueError):
        strict_json('{"a":NaN}')
    assert not grounded("oil", "coil failed")
    assert grounded("OIL leak", "inspect oil    leak now")


def evaluate(raw):
    row = {"id": "g:1", "text": "Repair the oil pump", "target": target(physical_objects=["oil pump"])}
    result = {"raw": raw, "input_tokens": 6, "generated_tokens": 12, "generation_seconds": 0.1}
    return score(row, result)


def test_micro_invalid_retained_and_match_normalized():
    good = evaluate(json.dumps(target(physical_objects=["OIL Pump", "oil  pump"])))
    invalid = evaluate("not JSON")
    wrong_shape = evaluate("[]")
    summary = summarize([good, invalid, wrong_shape])
    assert summary["count"] == 3
    assert summary["json_valid_rate"] == pytest.approx(2 / 3)
    assert summary["schema_compliance_rate"] == pytest.approx(1 / 3)
    assert summary["micro"]["f1"] == 0.5
    assert summary["fields"]["physical_objects"]["support"] == 3
    assert summary["macro_f1"] == 0.1


def test_boundary_and_hallucination():
    match = evaluate(json.dumps(target(physical_objects=["pump", "carburetor"])))
    assert {"boundary_candidate", "ungrounded_entity", "missing_entity"} <= set(match["errors"])
    assert match["score"]["fp"] == 2
