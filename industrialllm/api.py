"""本地模型 API。一个进程加载一次 Adapter，同一时刻执行一次生成。"""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from .model import Engine, InputLimitError
from .schema import Extraction, config, grounded, strict_json


class Ticket(BaseModel):
    text: str = Field(min_length=1, max_length=4096)

    @field_validator("text")
    @classmethod
    def has_text(cls, value):
        if not value.strip():
            raise ValueError("Ticket text is blank")
        return value


def real_engine():
    path = os.getenv("INDUSTRIALLLM_CONFIG", "configs/industrialllm.toml")
    manifest = os.getenv("INDUSTRIALLLM_MANIFEST", "outputs/standard/release_manifest.json")
    return Engine(config(path), os.getenv("INDUSTRIALLLM_DEVICE", "auto"), Path(manifest))


def create_app(engine_factory=real_engine):
    @asynccontextmanager
    async def lifespan(application):
        application.state.engine = engine_factory()
        application.state.generation_lock = Lock()
        yield

    application = FastAPI(title="IndustrialLLM", lifespan=lifespan)

    @application.get("/health")
    def health():
        return {"status": "ready"}

    @application.get("/models")
    def models(request: Request):
        engine = request.app.state.engine
        return {"id": engine.options["id"], "revision": engine.options["revision"],
                "version": engine.version, "profile": engine.profile, "device": engine.device}

    @application.post("/extract")
    def extract(ticket: Ticket, request: Request):
        engine = request.app.state.engine
        start = time.perf_counter()
        response = {
            "text": ticket.text, "model_version": engine.version, "result": None,
            "schema_valid": False, "grounded": False, "generation_ms": None, "error": None,
        }
        try:
            with request.app.state.generation_lock:
                result = engine.predict(ticket.text)
        except InputLimitError:
            response["error"] = "input_token_limit"
            status = 422
        else:
            response["generation_ms"] = result["generation_seconds"] * 1000
            try:
                prediction = Extraction.model_validate(strict_json(result["raw"])).model_dump()
            except ValueError:
                response["error"] = "invalid_model_output"
                status = 502
            else:
                response["schema_valid"] = True
                if any(not grounded(span, ticket.text) for spans in prediction.values() for span in spans):
                    response["error"] = "ungrounded_model_output"
                    status = 502
                else:
                    response.update(result=prediction, grounded=True)
                    status = 200
        response["request_ms"] = (time.perf_counter() - start) * 1000
        return JSONResponse(response, status_code=status)

    return application


app = create_app()
