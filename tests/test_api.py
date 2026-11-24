import json

from fastapi.testclient import TestClient

from industrialllm.api import create_app
from industrialllm.model import InputLimitError
from industrialllm.schema import FIELDS
from industrialllm.workflow import review


class Fake:
    version = "test-adapter"
    profile = "smoke"
    device = "cpu"
    options = {"id": "test-model", "revision": "fake"}

    def predict(self, text):
        if text == "long":
            raise InputLimitError("token limit")
        result = dict.fromkeys(FIELDS, []) | {"physical_objects": ["valve"]}
        return {"raw": "oops" if text == "bad" else json.dumps(result), "generation_seconds": 0.012}


def test_lifespan_and_three_routes():
    calls = []

    def factory():
        calls.append(True)
        return Fake()

    with TestClient(create_app(factory)) as client:
        assert client.get("/health").json()["status"] == "ready"
        assert client.get("/models").json()["version"] == "test-adapter"
        resp = client.post("/extract", json={"text": "replace valve"})
        assert resp.status_code == 200 and resp.json()["result"]["physical_objects"] == ["valve"]
        assert resp.json()["request_ms"] >= 0
        assert client.post("/extract", json={"text": "  "}).status_code == 422
        assert client.post("/extract", json={"text": "long"}).json()["error"] == "input_token_limit"
        assert client.post("/extract", json={"text": "bad"}).json()["error"] == "invalid_model_output"
        resp = client.post("/extract", json={"text": "replace pump"})
        assert resp.status_code == 502 and resp.json()["error"] == "ungrounded_model_output"
    assert calls == [True]


def test_review_records_human_decision():
    draft = {"status": "pending_review", "extraction": {
        "text": "replace valve", "result": dict.fromkeys(FIELDS, []) | {"physical_objects": ["valve"]},
    }}
    audited = review(draft, "approved", "Nan")
    assert audited["status"] == "approved" and audited["reviewer"] == "Nan"
    assert draft["status"] == "pending_review"
