# QA Service — RAG Document Q&A

A containerized FastAPI service that ingests PDF documents, answers natural
language questions with citations using Retrieval-Augmented Generation, and
evaluates answer quality with an LLM-as-judge.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  qa-service  (FastAPI :8000)                         │
│   /ingest  /query  /documents  /evaluate  /metrics   │
│              │                                       │
│              ├── sentence-transformers (local)       │
│              ├── NVIDIA NIM  (LLM, OpenAI-compat)    │
│              └── ChromaDB    (persistent on-disk)    │
└──────────────────────────────────────────────────────┘
```

Each ingested PDF is stored in its own Chroma collection so a single document
acts as an independently queryable "vector database". Queries can target a
specific document via `doc_id`, or fan out across every indexed document.

## Quickstart (Docker Compose)

```bash
cd assignments/qa-service
cp .env.example .env
# Edit .env and set LLM_API_KEY (free key at https://build.nvidia.com/)
docker compose up --build
```

Service is then live on `http://localhost:8000` with these endpoints:

| Method | Path        | Description                                       |
|--------|-------------|---------------------------------------------------|
| POST   | `/ingest`   | Multipart upload of a PDF, indexed into Chroma    |
| POST   | `/query`    | Ask a question, get answer + citations            |
| GET    | `/documents`| List indexed documents                            |
| POST   | `/evaluate` | Score generated answers against expected answers  |
| GET    | `/metrics`  | Prometheus exposition (`documents_ingested_total`, `queries_processed_total`) |
| GET    | `/health`   | Liveness probe                                    |

OpenAPI docs are served at `http://localhost:8000/docs`.

## Configuration

All settings come from environment variables (see `.env.example`):

| Variable              | Default                                            | Purpose                          |
|-----------------------|----------------------------------------------------|----------------------------------|
| `LLM_PROVIDER`        | `nvidia`                                           | Selector for the LLM provider    |
| `LLM_BASE_URL`        | `https://integrate.api.nvidia.com/v1`              | OpenAI-compatible base URL       |
| `LLM_MODEL`           | `meta/llama-3.1-8b-instruct`                       | Chat model name                  |
| `LLM_API_KEY`         | _(required)_                                       | NVIDIA API key (free tier)       |
| `LLM_TEMPERATURE`     | `0.1`                                              | Sampling temperature             |
| `LLM_MAX_TOKENS`      | `512`                                              | Response cap                     |
| `EMBEDDING_PROVIDER`  | `sentence-transformers`                            | Embedding backend                |
| `EMBEDDING_MODEL`     | `sentence-transformers/all-MiniLM-L6-v2`           | Embedding model                  |
| `VECTOR_STORE_URL`    | `/app/.chroma` (Docker) / `./.chroma` (local)      | Chroma persistent dir or HTTP endpoint |
| `CHUNK_SIZE`          | `512`                                              | Tokens per chunk                 |
| `CHUNK_OVERLAP`       | `50`                                               | Token overlap between chunks     |
| `TOP_K`               | `4`                                                | Retrieved chunks per query       |
| `LOG_LEVEL`           | `INFO`                                             | structlog level                  |
| `APP_PORT`            | `8000`                                             | Server port                      |

### Swapping providers

The LLM goes through any OpenAI-compatible endpoint. Examples:

| Provider | `LLM_BASE_URL`                              | Notes                              |
|----------|---------------------------------------------|------------------------------------|
| NVIDIA   | `https://integrate.api.nvidia.com/v1`       | Free tier on `build.nvidia.com`    |
| OpenAI   | `https://api.openai.com/v1`                 | Set `LLM_MODEL=gpt-4o-mini` etc.   |
| Ollama   | `http://host.docker.internal:11434/v1`      | Local, no key (use `not-needed`)   |

The embedding provider is currently sentence-transformers only; adding a new
provider is a small adapter at `providers/embedding.py`.

## Endpoint Examples

### Ingest a PDF

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@/path/to/handbook.pdf" \
  -F "title=Company Handbook"
```

Response:
```json
{
  "doc_id": "5b3a...-uuid",
  "title": "Company Handbook",
  "chunks_indexed": 12,
  "status": "success"
}
```

### Query a document

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the leave policy?"}'
```

Response:
```json
{
  "answer": "Employees are entitled to 18 days of paid leave...",
  "sources": [
    {"doc_id": "uuid", "title": "Company Handbook", "chunk": "..."}
  ]
}
```

Pass `"doc_id": "<uuid>"` to scope the query to one document.

### List documents

```bash
curl http://localhost:8000/documents
```

### Evaluate

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "test_cases": [
      {"question":"What is the notice period?","expected_answer":"30 days"}
    ]
  }'
```

### Metrics

```bash
curl http://localhost:8000/metrics | grep -E "documents_ingested_total|queries_processed_total"
```

## Local development (without Docker)

```bash
cd assignments/qa-service
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Use a local persistent Chroma directory instead of HTTP:
echo "VECTOR_STORE_URL=./.chroma" >> .env
echo "LLM_API_KEY=<your-nvidia-key>" >> .env

uvicorn qa_service.main:app --reload --port 8000
```

## Project Layout

```
qa-service/
├── Dockerfile                 # multi-stage slim runtime
├── docker-compose.yml         # qa-service (persistent on-disk Chroma)
├── .env.example               # all env vars documented
├── pyproject.toml             # dependencies + tooling config
├── src/qa_service/
│   ├── main.py                # FastAPI app
│   ├── config.py              # pydantic-settings
│   ├── dependencies.py        # DI singletons
│   ├── models/                # request/response models
│   ├── routes/                # API endpoints
│   ├── services/              # rag, judge, chunker, pdf_parser
│   ├── providers/             # embedding + LLM adapters
│   ├── repository/            # Chroma vector store
│   ├── observability/         # structlog + Prometheus
│   └── prompts/               # YAML prompt templates
└── docs/                      # generated documentation
```
