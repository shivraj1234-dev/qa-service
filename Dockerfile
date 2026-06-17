FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip \
    && pip install --prefix=/install .

# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/app/.hf_cache \
    PATH=/install/bin:$PATH

RUN useradd --create-home --uid 1001 qa

WORKDIR /app

COPY --from=builder /install /install
COPY src ./src

RUN mkdir -p /app/.hf_cache /app/.chroma && chown -R qa:qa /app

USER qa

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "qa_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
