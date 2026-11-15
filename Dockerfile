FROM python:3.13-slim AS build

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project


FROM python:3.13-slim

RUN useradd --system --uid 10001 --home-dir /app app

WORKDIR /app
COPY --from=build /app/.venv .venv
COPY main.py ./
COPY app ./app
COPY data ./data
# data/tiles is where `pipeline tiles` keeps the NASA photo tiles; mount a volume there to keep them.
RUN mkdir -p data/tiles && chown -R app:app data

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000

USER app
EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn main:app --host \"$HOST\" --port \"$PORT\""]
