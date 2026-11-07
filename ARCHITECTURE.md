# 🏗️ System Architecture & Technical Design

The complete, diagram-rich architecture and technical design documentation for this project is available in:

👉 **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**

It includes:
- **High-Level System Architecture Diagram** (Ingestion, Hybrid Retrieval, Prompt Augmentation, Generation, Interfaces)
- **Directory Structure & Component Matrix** (Responsibilities and dependencies for every file)
- **Pydantic Data Schemas & Class Diagram** (`MinimalSource`, `MinimalAnswer`, etc.)
- **Ingestion & Language-Aware Chunking Pipeline Diagram** (Python AST, Markdown headers, Text splitting & offset invariant repair)
- **Incremental Indexing State Diagram** (Manifest caching, `mtime` & SHA-256 dirty checking, dual BM25 + ChromaDB upsert)
- **Hybrid Retrieval & Score Fusion Diagram** (BM25 normalization, ChromaDB distance conversion, WSF & RRF)
- **Context Augmentation & Generation Pipeline Diagram** (On-demand text slicing, 15k char token budget packing, deterministic Qwen3-0.6B local inference, caching)
- **Ground-Truth Evaluation & IoU Calculation Diagram** (IoU $\ge 0.05$ calculation and Recall@k validation)
- **End-to-End Sequence Diagram** (Client request lifecycle through FastAPI / CLI to LLM response)
- **Performance Benchmarks & Key Design Decisions** (Recall@5 scores, indexing latency, trade-off analysis)
