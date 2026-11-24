FROM ghcr.io/astral-sh/uv:0.13.7 AS uv
FROM python:3.12-slim-bookworm
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never PYTHONUNBUFFERED=1 \
    INDUSTRIALLLM_DEVICE=cpu INDUSTRIALLLM_MANIFEST=/artifacts/release_manifest.json \
    HF_HOME=/hf HF_HUB_OFFLINE=1
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --extra cpu --extra model
COPY industrialllm industrialllm
COPY configs configs
EXPOSE 8000
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "industrialllm.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
