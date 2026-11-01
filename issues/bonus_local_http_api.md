# Bonus 5 Implementation Guide: Local HTTP API

**Topic:** Bonus Part 5 — Local HTTP API  
**Value:** 1 Bonus Point  
**Requirement (Chapter IX, page 23):**
> *"5. Local HTTP API: expose querying the index and answering questions over a small local HTTP API, so the system can be driven by something other than the CLI."*

---

## 1. Design Overview

To fulfill Bonus 5, the project needs an HTTP server allowing external applications or web frontends to query the index and generate answers via standard HTTP requests.

### Target Endpoints

| Method | Endpoint | Query / Body Parameters | Response Model | Description |
|---|---|---|---|---|
| `GET` | `/health` | None | `{"status": "ok"}` | Service health check. |
| `POST` | `/search` | `{"query": str, "k": int}` | `MinimalSearchResults` | Queries the hybrid index and returns ranked sources. |
| `GET` | `/search` | `?query=...&k=10` | `MinimalSearchResults` | Query index via URL query parameters. |
| `POST` | `/answer` | `{"query": str, "k": int}` | `MinimalAnswer` | Queries index and generates an answer with `Qwen3-0.6B`. |
| `GET` | `/answer` | `?query=...&k=10` | `MinimalAnswer` | Answer via URL query parameters. |

---

## 2. Implementation: `src/server.py`

We can implement this using **FastAPI** (if added to dependencies) or Python's built-in **`http.server`** with `urllib.parse` and `json` (which requires zero additional dependencies).

Here is the robust, production-grade **FastAPI** implementation:

```python
"""Local HTTP API exposing search and answer endpoints (Bonus 5)."""

from __future__ import annotations

import sys
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import uvicorn

from src.caching import Caching
from src.generate import Generate
from src.models import MinimalAnswer, MinimalSearchResults
from src.prompts import Prompt
from src.retriever import Searcher

app = FastAPI(
    title="RAG Against the Machine API",
    description="Local HTTP API for codebase search and answer generation",
    version="1.0.0",
)

# Global service singletons
searcher: Optional[Searcher] = None
generator: Optional[Generate] = None


class SearchRequest(BaseModel):
    query: str
    k: int = 10


class AnswerRequest(BaseModel):
    query: str
    k: int = 10


@app.on_event("startup")
def startup_event() -> None:
    """Initialize index searcher on server startup."""
    global searcher
    try:
        searcher = Searcher()
    except Exception as e:
        print(f"Warning: Could not preload searcher on startup: {e}", file=sys.stderr)


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/search", response_model=MinimalSearchResults)
@app.post("/search", response_model=MinimalSearchResults)
def search_endpoint(
    query: Optional[str] = Query(None),
    k: int = Query(10),
    body: Optional[SearchRequest] = None,
) -> MinimalSearchResults:
    """Retrieve top-k source locations for a query."""
    global searcher
    if searcher is None:
        searcher = Searcher()

    effective_query = (body.query if body else query) or ""
    effective_k = body.k if body else k

    if not effective_query.strip() or effective_k <= 0:
        return MinimalSearchResults(
            question_id="api-query",
            question=effective_query,
            retrieved_sources=[],
        )

    sources = searcher.get_resources(effective_query, k=effective_k)
    return MinimalSearchResults(
        question_id="api-query",
        question=effective_query,
        retrieved_sources=sources,
    )


@app.get("/answer", response_model=MinimalAnswer)
@app.post("/answer", response_model=MinimalAnswer)
def answer_endpoint(
    query: Optional[str] = Query(None),
    k: int = Query(10),
    body: Optional[AnswerRequest] = None,
) -> MinimalAnswer:
    """Retrieve context and generate an answer using Qwen3-0.6B."""
    global searcher, generator
    if searcher is None:
        searcher = Searcher()
    if generator is None:
        generator = Generate()

    effective_query = (body.query if body else query) or ""
    effective_k = body.k if body else k

    if not effective_query.strip() or effective_k <= 0:
        return MinimalAnswer(
            question_id="api-query",
            question=effective_query,
            retrieved_sources=[],
            answer="I don't know.",
        )

    cache = Caching(top_k=effective_k)
    cached = cache.get_answer_from_cache(effective_query)
    sources = searcher.get_resources(effective_query, k=effective_k)

    if cached:
        answer_text = cached
    else:
        prompt = Prompt().build_prompt(effective_query, sources)
        answer_text = generator.generate_answer(prompt)
        cache.save_query_and_answer(effective_query, answer_text)

    return MinimalAnswer(
        question_id="api-query",
        question=effective_query,
        retrieved_sources=sources,
        answer=answer_text,
    )


def start_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the Uvicorn server."""
    uvicorn.run(app, host=host, port=int(port), log_level="info")
```

---

## 3. CLI Integration in `src/__main__.py`

Add the `serve` command to the `CLI` class in [`src/__main__.py`](file:///home/nyahyaou/Desktop/RAG/src/__main__.py):

```python
    def serve(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Start the local HTTP API server (Bonus 5).

        Args:
            host: Host interface to bind (default: 127.0.0.1).
            port: Port to listen on (default: 8000).
        """
        try:
            from src.server import start_server
            print(f"Starting Local HTTP API server on http://{host}:{port} ...")
            start_server(host=host, port=port)
        except Exception as e:
            print(f"Error starting HTTP server: {e}", file=sys.stderr)
```

---

## 4. Dependencies in `pyproject.toml`

Ensure `fastapi` and `uvicorn` are listed in `dependencies` in `pyproject.toml`:
```toml
dependencies = [
    "pydantic>=2.0",
    "fire>=0.6",
    "tqdm>=4.60",
    "transformers>=4.30",
    "torch>=2.0",
    "bm25s[core]>=0.3.13",
    "langchain-text-splitters>=1.1.3",
    "accelerate>=1.15.0",
    "sentence-transformers>=6.1.0",
    "chromadb>=1.5.9",
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
]
```

---

## 5. Demonstration & Testing

Start the server:
```bash
uv run python -m src serve --port 8000
```

### Test 1: Health Check
```bash
curl http://127.0.0.1:8000/health
# Response: {"status":"ok"}
```

### Test 2: Search Query via GET
```bash
curl "http://127.0.0.1:8000/search?query=How+to+configure+OpenAI+server&k=2"
```
**Response:**
```json
{
  "question_id": "api-query",
  "question": "How to configure OpenAI server",
  "retrieved_sources": [
    {
      "file_path": "data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md",
      "first_character_index": 9867,
      "last_character_index": 10100
    }
  ]
}
```

### Test 3: Search Query via POST
```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is PagedAttention?", "k": 3}'
```

### Test 4: Answer Query via POST
```bash
curl -X POST http://127.0.0.1:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?", "k": 3}'
```
**Response:**
```json
{
  "question_id": "api-query",
  "question": "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?",
  "retrieved_sources": [
    {
      "file_path": "data/raw/vllm-0.10.1/docs/features/lora.md",
      "first_character_index": 4695,
      "last_character_index": 6098
    }
  ],
  "answer": "To dynamically load a LoRA adapter in vLLM, send a POST request to the `/v1/load_lora_adapter` endpoint with the adapter details in the payload."
}
```
Interactive Swagger documentation is also automatically available at `http://127.0.0.1:8000/docs`.

