# 🏗️ System Architecture & Technical Design

> **Project**: `rag-against-the-machine` — Production-Grade Retrieval-Augmented Generation (RAG) System for the **vLLM Codebase**  
> **Author**: `nyahyaou` (42 Curriculum)

---

## Table of Contents
1. [Executive Summary & High-Level Architecture](#1-executive-summary--high-level-architecture)
2. [Project Structure & Component Matrix](#2-project-structure--component-matrix)
3. [Data Models & Schema Architecture](#3-data-models--schema-architecture)
4. [Subsystem 1: Ingestion & Language-Aware Chunking](#4-subsystem-1-ingestion--language-aware-chunking)
5. [Subsystem 2: Incremental Indexing & Dual-Storage Engine](#5-subsystem-2-incremental-indexing--dual-storage-engine)
6. [Subsystem 3: Hybrid Retrieval & Score Fusion](#6-subsystem-3-hybrid-retrieval--score-fusion)
7. [Subsystem 4: Context Augmentation & Local LLM Generation](#7-subsystem-4-context-augmentation--local-llm-generation)
8. [Subsystem 5: Ground-Truth Evaluation & Span IoU Verification](#8-subsystem-5-ground-truth-evaluation--span-iou-verification)
9. [Subsystem 6: Service Interfaces & End-to-End Execution Flow](#9-subsystem-6-service-interfaces--end-to-end-execution-flow)
10. [Performance Benchmarks & Key Design Decisions](#10-performance-benchmarks--key-design-decisions)

---

## 1. Executive Summary & High-Level Architecture

Large Language Models (LLMs) frequently hallucinate or generate outdated information when queried about complex, rapidly evolving software repositories such as [vLLM](https://github.com/vllm-project/vllm).

**RAG Against the Machine** solves this problem by decoupling knowledge retrieval from language generation. The architecture comprises a four-stage pipeline:

```mermaid
flowchart TD
    subgraph S1["1. Ingestion & Indexing Pipeline"]
        direction TB
        RAW["📁 Raw Codebase (data/raw/vllm-0.10.1/)"] --> SCAN["🔍 scanner.py (Recursive File Discovery)"]
        SCAN --> ROUTE{"Language Router"}
        ROUTE -->|".py"| CHK_PY["✂️ python_chunker.py (AST / Def-Class Split)"]
        ROUTE -->|".md"| CHK_MD["✂️ markdown_chunker.py (Header-Aware Split)"]
        ROUTE -->|".txt"| CHK_TXT["✂️ text_chunker.py (Paragraph / Recursive Split)"]
        CHK_PY & CHK_MD & CHK_TXT --> VAL["🛡️ base.py (Offset Slicing Validator)"]
        VAL --> MANIFEST["📋 manifest.json (mtime & SHA-256 Change Tracking)"]
        MANIFEST --> IDX["📊 indexer.py (Dual Index Builder)"]
        IDX --> BM25_STORE["💾 BM25 Lexical Index (lexical_index.pkl)"]
        IDX --> CHROMA_STORE["💾 ChromaDB Vector Store (semantic_index Collection)"]
    end

    subgraph S2["2. Hybrid Retrieval & Score Fusion"]
        direction TB
        QUERY["❓ User Query / Dataset Prompt"] --> RET["🔎 retriever.py (Searcher)"]
        BM25_STORE -->|"BM25 Candidates (Top 3*k)"| RET
        CHROMA_STORE -->|"Vector Candidates (Top 3*k)"| RET
        RET --> FUSION{"Score Fusion Strategy"}
        FUSION -->|"WSF Mode (Default)"| WSF["⚖️ Weighted Score Fusion (WSF)\nLexical: 0.8 | Semantic: 0.2"]
        FUSION -->|"RRF Mode"| RRF["📊 Reciprocal Rank Fusion (RRF)\nConstant k=60"]
        WSF & RRF --> TOPK["📦 Ranked Top-K MinimalSource Spans"]
    end

    subgraph S3["3. Context Augmentation & Generation"]
        direction TB
        TOPK --> BUDGET["🧱 prompts.py (Prompt Builder & Context Budget)"]
        QUERY --> BUDGET
        BUDGET -->|"Context <= 15,000 Chars"| GEN["🤖 generate.py (Qwen3-0.6B LLM)"]
        CACHE["⚡ caching.py (query_cache.json)"] <-->|"Cache Check & Warm Store"| GEN
        GEN --> ANS["✅ Final Answer (MinimalAnswer JSON)"]
    end

    subgraph S4["4. User & Service Interfaces"]
        direction LR
        CLI["💻 CLI Entrypoint (src/__main__.py)"]
        API["🌐 REST API (src/server.py via FastAPI)"]
    end

    S4 -.-> S2
    S4 -.-> S3

    style S1 fill:#f0f7ff,stroke:#0284c7,stroke-width:2px
    style S2 fill:#fffbeb,stroke:#d97706,stroke-width:2px
    style S3 fill:#f0fdf4,stroke:#16a34a,stroke-width:2px
    style S4 fill:#faf5ff,stroke:#9333ea,stroke-width:2px
```

---

## 2. Project Structure & Component Matrix

### 2.1 File System Tree

```
RAG/
├── src/                          # Core application source code
│   ├── __init__.py               # Package initializer
│   ├── __main__.py               # CLI entry point using Python Fire
│   ├── config.py                 # Central configurations & system constants
│   ├── models.py                 # Pydantic schemas and data validation
│   ├── scanner.py                # Raw file discovery and categorization
│   ├── indexer.py                # Dual index builder (BM25 + ChromaDB) & manifest manager
│   ├── retriever.py              # Hybrid retrieval engine (WSF & RRF fusion)
│   ├── prompts.py                # Dynamic disk loader & context budget truncator
│   ├── generate.py               # Deterministic local LLM generator (Qwen3-0.6B)
│   ├── caching.py                # Normalized query/answer persistent disk cache
│   ├── evaluator.py              # Recall@k & character-span IoU evaluation engine
│   ├── server.py                 # FastAPI HTTP service (/search and /answer)
│   └── chunking/                 # Language-specific chunking sub-package
│       ├── __init__.py           # Sub-package exports
│       ├── base.py               # Strict chunk invariant validator
│       ├── python_chunker.py     # Python AST / syntax-aware splitter
│       ├── markdown_chunker.py   # Markdown structural header splitter
│       └── text_chunker.py       # Generic whitespace and paragraph splitter
├── data/
│   ├── raw/vllm-0.10.1/          # Target raw codebase (.py, .md, .txt)
│   ├── processed/                # Built indices (BM25 pkl, ChromaDB SQLite, manifest.json)
│   ├── datasets/                 # Reference evaluation datasets (Answered & Unanswered)
│   ├── cache/                    # query_cache.json
│   └── output/                   # Evaluation runs, student outputs & answer files
├── docs/
│   ├── ARCHITECTURE.md           # This comprehensive architecture document
│   └── OPTIMIZATION_GUIDE.md     # Performance tuning and indexing benchmarks
├── pyproject.toml                # Project metadata, dependencies and tools
├── uv.lock                       # Deterministic dependency lockfile
├── Makefile                      # Standard automation recipes (install, lint, run)
└── README.md                     # High-level overview & evaluation scorecard
```

### 2.2 Component Responsibility Matrix

| Module | Core Class / Method | Key Responsibility | Primary Dependencies |
| :--- | :--- | :--- | :--- |
| [`src/config.py`](src/config.py) | Constants | Single source of truth for paths, thresholds, budget sizes, and model IDs | `pathlib.Path` |
| [`src/models.py`](src/models.py) | `MinimalSource`, `MinimalAnswer` | Pydantic data contract validation across all system boundaries | `pydantic`, `uuid` |
| [`src/scanner.py`](src/scanner.py) | `scan_files()` | Recursively discovers files in `data/raw/` grouped by extension (`.py`, `.md`, `.txt`) | `pathlib.Path` |
| [`src/chunking/`](src/chunking/) | `chunk_*_file()`, `validate_chunk()` | Splits files preserving character boundaries and verifies slicing invariant: `content[start:end] == text` | `langchain_text_splitters` |
| [`src/indexer.py`](src/indexer.py) | `Index` | Manages incremental indexing, dirty file detection, BM25 pickling, and ChromaDB vector upserting | `bm25s`, `PyStemmer`, `SentenceTransformer`, `chromadb` |
| [`src/retriever.py`](src/retriever.py) | `Searcher` | Executes lexical search and semantic search, combines candidate lists using WSF or RRF | `bm25s`, `chromadb`, `numpy` |
| [`src/prompts.py`](src/prompts.py) | `Prompt` | Loads chunk text dynamically from disk offsets and enforces character budget ($\le 15,000$ chars) | Builtin string ops |
| [`src/generate.py`](src/generate.py) | `Generate` | Local deterministic inference with `Qwen/Qwen3-0.6B` (`temperature=0.0`, greedy decode) | `transformers`, `torch` |
| [`src/caching.py`](src/caching.py) | `Caching` | Normalizes questions, handles warm cache lookups, and persists responses to disk | `json`, `pathlib` |
| [`src/evaluator.py`](src/evaluator.py) | `Evaluator` | Computes Intersection over Union ($\text{IoU} \ge 0.05$) and calculates Recall@1, 3, 5, 10 | `json`, `pydantic` |
| [`src/server.py`](src/server.py) | FastAPI app | Exposes REST HTTP endpoints (`GET /search`, `GET /answer`) | `fastapi`, `uvicorn` |
| [`src/__main__.py`](src/__main__.py) | CLI | Exposes command-line interface via Python Fire for all pipeline actions | `fire` |

---

## 3. Data Models & Schema Architecture

All data flowing through RAG Against the Machine is strongly typed and validated by Pydantic models in [`src/models.py`](src/models.py). This enforces character offset integrity from ingestion to response serialization.

```mermaid
classDiagram
    class MinimalSource {
        +UUID|str id
        +str text
        +str file_path
        +int first_character_index
        +int last_character_index
    }

    class UnansweredQuestion {
        +str question_id
        +str question
    }

    class AnsweredQuestion {
        +List~MinimalSource~ sources
        +str answer
    }

    class MinimalSearchResults {
        +str question_id
        +str question
        +List~MinimalSource~ retrieved_sources
    }

    class MinimalAnswer {
        +str answer
    }

    class StudentSearchResults {
        +List~MinimalSearchResults~ search_results
        +int k
    }

    class StudentSearchResultsAndAnswer {
        +List~MinimalAnswer~ search_results
        +int k
    }

    class RagDataset {
        +List~AnsweredQuestion|UnansweredQuestion~ rag_questions
    }

    UnansweredQuestion <|-- AnsweredQuestion : Extends
    MinimalSearchResults <|-- MinimalAnswer : Extends
    StudentSearchResults *-- MinimalSearchResults : Aggregates
    StudentSearchResultsAndAnswer *-- MinimalAnswer : Aggregates
    MinimalSearchResults o-- MinimalSource : References
    AnsweredQuestion o-- MinimalSource : References
    RagDataset o-- UnansweredQuestion : Contains
```

### Core Schema Invariant: `MinimalSource`
The `MinimalSource` object represents an exact character span within a target file:
- `file_path`: Path to the source file relative to repository root.
- `first_character_index`: Start character index (0-indexed, inclusive).
- `last_character_index`: End character index (exclusive, Python slice semantics).
- `text`: Chunk text string matching `raw_file[first:last]`.

---

## 4. Subsystem 1: Ingestion & Language-Aware Chunking

The ingestion pipeline transforms raw code and documentation into bounded, semantically coherent text chunks ($\le 2,000$ characters) while strictly maintaining character-level alignment.

```mermaid
flowchart TD
    START["📄 Raw File on Disk"] --> SCAN["scanner.py\nscan_files()"]
    SCAN --> ROUTE{"File Extension?"}
    
    ROUTE -->|".py"| PY["chunk_python_file()\nLangChain RecursiveCharacterTextSplitter\nLanguage.PYTHON (def, class, comments)"]
    ROUTE -->|".md"| MD["chunk_markdown_file()\nLangChain RecursiveCharacterTextSplitter\nLanguage.MARKDOWN (#, ##, ### headers)"]
    ROUTE -->|".txt"| TXT["chunk_text_file()\nRecursiveCharacterTextSplitter\nGeneric paragraphs & whitespace"]

    PY & MD & TXT --> METADATA["Extract metadata['start_index']"]
    METADATA --> OFFSET_CHECK{"content[start:end] == text?"}
    
    OFFSET_CHECK -->|"Yes"| CHUNK_OBJ["Instantiate MinimalSource"]
    OFFSET_CHECK -->|"No"| REPAIR["content.find(text, start)\nRealign exact start & end indices"]
    REPAIR --> CHUNK_OBJ

    CHUNK_OBJ --> VALIDATE{"validate_chunk()\n- last - first == len(text)\n- len(text) <= max_chunk_size\n- content[first:last] == text\n- 0 <= first <= last <= len(file)"}
    
    VALIDATE -->|"Valid"| OUT["Emit MinimalSource"]
    VALIDATE -->|"Invalid"| DROP["Discard / Raise Error"]

    style START fill:#f1f5f9,stroke:#475569
    style VALIDATE fill:#ecfdf5,stroke:#059669,stroke-width:2px
    style REPAIR fill:#fff7ed,stroke:#c2410c
    style OUT fill:#dbeafe,stroke:#1d4ed8
```

### Why Offset Alignment Matters
1. **Evaluation Requirement**: The grading harness validates candidate chunks against ground-truth spans using character Intersection over Union ($\text{IoU} \ge 0.05$). Offsets that drift even slightly will fail the IoU test.
2. **Dynamic Context Loading**: [`src/prompts.py`](src/prompts.py) reads the text directly from disk using `content[first:last]` when assembling LLM prompts to conserve RAM during indexing.

---

## 5. Subsystem 2: Incremental Indexing & Dual-Storage Engine

To prevent costly re-computation on large codebases, [`src/indexer.py`](src/indexer.py) implements **incremental indexing** using a file change tracking manifest (`manifest.json`).

```mermaid
flowchart TD
    START["Run: python -m src index"] --> LOAD_MAN["Load manifest.json\n(mtime, SHA-256 hash, cached chunks)"]
    LOAD_MAN --> SCAN_DISK["Scan all current disk files (all_files)"]
    
    SCAN_DISK --> PURGE["Purge Deleted Files\nmanifest_files - current_files"]
    PURGE --> DEL_CHROMA["ChromaDB.delete(where={'file_path': deleted})"]
    DEL_CHROMA --> DEL_MAN["manifest.pop(deleted)"]
    
    DEL_MAN --> LOOP["Iterate over each file_path"]
    LOOP --> CHECK_MTIME{"Is mtime identical?"}
    
    CHECK_MTIME -->|"Yes (Clean)"| REUSE["Reuse cached chunks from manifest"]
    CHECK_MTIME -->|"No (Potentially Dirty)"| CALC_HASH["Compute SHA-256 Checksum"]
    
    CALC_HASH --> CHECK_HASH{"Is SHA-256 identical?"}
    CHECK_HASH -->|"Yes (Content Unchanged)"| UPDATE_MTIME["Update manifest mtime only\nReuse cached chunks"]
    CHECK_HASH -->|"No (Content Changed)"| RECHUNK["1. ChromaDB.delete(where={'file_path': file})\n2. Run language chunker on file\n3. Save new chunks & hash to manifest"]
    
    RECHUNK --> APPEND_NEW["Add to new_chunks (for vector embedding)"]
    REUSE & UPDATE_MTIME --> APPEND_ALL["Add to all_chunks (for BM25 indexing)"]
    APPEND_NEW --> APPEND_ALL

    APPEND_ALL --> SAVE_MAN["Save manifest.json to disk"]
    SAVE_MAN --> BUILD_BM25["Build Lexical Index (all_chunks)\nTokenize with PyStemmer + Stopwords\nPickle to data/processed/lexical_index.pkl"]
    SAVE_MAN --> BUILD_VEC["Build Semantic Index (new_chunks ONLY)\nSentenceTransformer('all-MiniLM-L6-v2')\nUpsert to ChromaDB in batches of 5,000"]
    
    BUILD_BM25 & BUILD_VEC --> DONE["✅ Indexing Complete"]

    style CHECK_MTIME fill:#fef3c7,stroke:#d97706
    style CHECK_HASH fill:#fef3c7,stroke:#d97706
    style RECHUNK fill:#fee2e2,stroke:#b91c1c
    style DONE fill:#dcfce7,stroke:#15803d,stroke-width:2px
```

### Storage Architecture
- **Lexical Store (`data/processed/lexical_index.pkl`)**: Python pickle containing `IndexData`:
  - `bm25`: Fitted `bm25s.BM25` retrieval structure tokenized with English stopwords and the Snowball Stemmer (`PyStemmer`).
  - `chunks`: In-memory list of all `MinimalSource` chunk headers.
- **Dense Vector Store (`data/processed/`)**: Persistent ChromaDB instance storing the collection `"semantic_index"`:
  - Embeddings: 384-dimensional dense vectors generated by `all-MiniLM-L6-v2`.
  - Documents: Chunk text.
  - Metadatas: `file_path`, `first_character_index`, `last_character_index`.
- **Manifest (`data/processed/manifest.json`)**: JSON mapping from `file_path` to modification time, SHA-256 hash, and pre-computed chunk offset list.

---

## 6. Subsystem 3: Hybrid Retrieval & Score Fusion

Code queries often combine exact technical tokens (e.g. `SamplingParams`, `vllm.engine.async_llm_engine`) with semantic conceptual phrasing (e.g. "how to enable prefix caching?"). RAG Against the Machine executes **hybrid retrieval** by combining lexical BM25 and dense semantic embeddings.

```mermaid
flowchart TD
    Q["Query: 'How to configure OpenAI server?'"] --> RET["Searcher Engine"]
    
    subgraph LEX["Lexical Search Path (bm25s)"]
        direction TB
        T["Tokenize query\nStopwords + PyStemmer"] --> B_QUERY["BM25 Retrieve\nFetch top candidate chunks"]
        B_QUERY --> B_NORM["Min-Max Normalization\nS_lex = (s - s_min) / (s_max - s_min) * 0.8"]
    end

    subgraph SEM["Semantic Search Path (ChromaDB)"]
        direction TB
        E["Encode query\nall-MiniLM-L6-v2 (384-d, normalized)"] --> C_QUERY["ChromaDB Query\nCosine Distance d in [0, 2]"]
        C_QUERY --> C_NORM["Distance Inversion & Scaling\nS_sem = (1.0 - d / 2.0) * 0.2"]
    end

    RET --> LEX
    RET --> SEM

    B_NORM --> FUSION["Weighted Score Fusion (WSF)\nS_hybrid = S_lex + S_sem"]
    C_NORM --> FUSION

    FUSION --> DEDUP["Deduplicate by chunk ID & Sort Descending"]
    DEDUP --> SLICE["Slice [:k] Top Results"]
    SLICE --> OUT["list[MinimalSource]"]

    style LEX fill:#eff6ff,stroke:#2563eb
    style SEM fill:#fefce8,stroke:#ca8a04
    style FUSION fill:#ecfdf5,stroke:#059669,stroke-width:2px
```

### 6.1 Weighted Score Fusion (WSF) Formulation

1. **Lexical Score Normalization:**
   $$S_{\text{lex}}(c) = \begin{cases} 
   0.0 & \text{if } s_{\max} = s_{\min} \\
   \frac{s(c) - s_{\min}}{s_{\max} - s_{\min}} \times w_{\text{lex}} & \text{otherwise}
   \end{cases}$$
   Where $w_{\text{lex}} = 0.8$ (default).

2. **Semantic Score Conversion:**
   ChromaDB returns cosine distances $d \in [0, 2]$, where $0$ is an exact match and $2$ is diametrically opposite. To convert to a similarity score bounded in $[0, 1]$:
   $$S_{\text{sem}}(c) = \left(1.0 - \frac{d(c)}{2.0}\right) \times w_{\text{sem}}$$
   Where $w_{\text{sem}} = 0.2$ (default).

3. **Combined Hybrid Score:**
   $$S_{\text{hybrid}}(c) = S_{\text{lex}}(c) + S_{\text{sem}}(c)$$

### 6.2 Alternative: Reciprocal Rank Fusion (RRF)
The engine also supports rank-based fusion without score calibration:
$$\text{RRF}(c) = \sum_{r \in \{\text{lex}, \text{sem}\}} \frac{1}{60 + \text{rank}_r(c)}$$

---

## 7. Subsystem 4: Context Augmentation & Local LLM Generation

The generation subsystem formats the retrieved evidence into a token-budgeted prompt and generates a grounded response using a local language model.

```mermaid
flowchart TD
    Q["User Question"] --> CACHE_CHECK{"caching.py\nIs question in query_cache.json?"}
    
    CACHE_CHECK -->|"Hit (Warm Query)"| RETURN_CACHE["⚡ Return Cached Answer Immediately"]
    CACHE_CHECK -->|"Miss (Cold Query)"| RETRIEVE["retriever.py\nRetrieve Top-K MinimalSource chunks"]
    
    RETRIEVE --> LOAD_TEXT{"Does chunk have text in memory?"}
    LOAD_TEXT -->|"Yes"| USE_TEXT["Use chunk.text"]
    LOAD_TEXT -->|"No"| READ_DISK["prompts.py: load_chunk_text()\nSlice raw file: content[first:last]"]
    
    USE_TEXT & READ_DISK --> BUDGET["truncate_contexts()\nPack chunks until sum(len(text)) >= 15,000 chars"]
    
    BUDGET --> TEMPLATE["build_prompt()\nContext: {retrieved_context}\n\nQuestion: {question}"]
    
    TEMPLATE --> LLM["generate.py: Qwen/Qwen3-0.6B\n- device_map='auto'\n- do_sample=False (Greedy Decoding)\n- temperature=0.0\n- max_new_tokens=512"]
    
    LLM --> STORE_CACHE["caching.py: save_query_and_answer()\nPersist answer to data/cache/query_cache.json"]
    STORE_CACHE --> FINAL_OUT["Return Answer"]
    RETURN_CACHE --> FINAL_OUT

    style CACHE_CHECK fill:#fef3c7,stroke:#d97706
    style RETURN_CACHE fill:#dcfce7,stroke:#16a34a,stroke-width:2px
    style LLM fill:#e0e7ff,stroke:#4338ca,stroke-width:2px
```

### Deterministic Inference Configuration
- **Model**: `Qwen/Qwen3-0.6B`
- **Context Budget**: $15,000$ characters (guaranteeing accommodation within Qwen's 32,768 token window without context overflow).
- **Sampling Settings**: `do_sample=False`, `temperature=0.0` (zero randomness ensures 100% reproducible answers).
- **Generation Limit**: `max_new_tokens = 512`.
- **Query Cache Key**: `" ".join(query.lower().split()) + " [top_k={k}]"`.

---

## 8. Subsystem 5: Ground-Truth Evaluation & Span IoU Verification

The evaluation module ([`src/evaluator.py`](src/evaluator.py)) checks student retrieval accuracy against ground-truth spans using character-level **Intersection over Union (IoU)**.

```mermaid
flowchart TD
    STUDENT["Student Search Results\n(List of MinimalSource spans)"] --> EVAL["Evaluator Engine"]
    GROUND_TRUTH["Ground Truth Dataset\n(Reference MinimalSource spans)"] --> EVAL
    
    EVAL --> FILE_CHECK{"Same file_path?\nraw.file_path == student.file_path"}
    FILE_CHECK -->|"No"| INVALID["❌ Invalid Match (IoU = 0.0)"]
    
    FILE_CHECK -->|"Yes"| CALC_SPAN["Compute Span Overlaps:\nstart_nested = max(first_raw, first_student)\nend_nested = min(last_raw, last_student)\nIntersection = max(0, end_nested - start_nested)\nUnion = max(last_raw, last_student) - min(first_raw, first_student)"]
    
    CALC_SPAN --> IOU_CHECK{"Intersection / Union >= 0.05?"}
    IOU_CHECK -->|"Yes"| VALID["✅ Valid Match"]
    IOU_CHECK -->|"No"| INVALID

    VALID --> RECALL["Check Top-K:\nDoes any valid match appear in rank 1..k?"]
    RECALL --> METRICS["Compute Recall@k for k in [1, 3, 5, 10]\nRecall@k = sum(hits) / total_questions"]

    style FILE_CHECK fill:#fef3c7,stroke:#d97706
    style IOU_CHECK fill:#fef3c7,stroke:#d97706
    style VALID fill:#dcfce7,stroke:#15803d,stroke-width:2px
    style INVALID fill:#fee2e2,stroke:#b91c1c
    style METRICS fill:#dbeafe,stroke:#1e40af
```

### Mathematical IoU Formula
$$\text{IoU} = \frac{\max\Big(0, \ \min(\text{end}_{\text{ref}}, \text{end}_{\text{stu}}) - \max(\text{start}_{\text{ref}}, \text{start}_{\text{stu}})\Big)}{\max(\text{end}_{\text{ref}}, \text{end}_{\text{stu}}) - \min(\text{start}_{\text{ref}}, \text{start}_{\text{stu}})} \ge 0.05$$

A query is counted as a success at $k$ if **at least one** candidate in the top-$k$ returned sources has an $\text{IoU} \ge 0.05$ with the ground truth span.

---

## 9. Subsystem 6: Service Interfaces & End-to-End Execution Flow

The system supports two execution paths:
1. **Interactive CLI** (`src/__main__.py` via Python Fire)
2. **REST API Service** (`src/server.py` via FastAPI / Uvicorn)

### End-to-End Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor Client as User / HTTP Client
    participant Server as FastAPI / CLI
    participant Cache as Caching Engine
    participant Retriever as Searcher (retriever.py)
    participant BM25 as BM25 Index (bm25s)
    participant Chroma as ChromaDB Vector Store
    participant Prompts as Prompt Builder
    participant LLM as Qwen3-0.6B LLM

    Client->>Server: GET /answer?query="..."&k=10
    Server->>Cache: get_answer_from_cache(query)
    
    alt Cache Hit
        Cache-->>Server: Return cached answer string
        Server-->>Client: 200 OK {"answer": cached_answer}
    else Cache Miss
        Server->>Retriever: get_resources(query)
        par Lexical Retrieval
            Retriever->>BM25: BM25 retrieve candidates
            BM25-->>Retriever: Top-30 lexical hits + scores
        and Dense Vector Retrieval
            Retriever->>Chroma: ChromaDB query(vector, n_results=30)
            Chroma-->>Retriever: Top-30 vector hits + cosine distances
        end
        Retriever->>Retriever: Weighted Score Fusion (WSF: 0.8 / 0.2)
        Retriever-->>Server: Top-10 MinimalSource objects
        
        Server->>Prompts: build_prompt(query, sources)
        Prompts->>Prompts: Truncate contexts to budget (<= 15,000 chars)
        Prompts-->>Server: Formatted grounded prompt string
        
        Server->>LLM: generate_answer(prompt)
        LLM-->>Server: Generated answer text
        
        Server->>Cache: save_query_and_answer(query, answer)
        Server-->>Client: 200 OK {"answer": answer, "sources": [...]}
    end
```

### CLI Command Reference
- **Build Index**: `uv run python -m src index --max_chunk_size 2000`
- **Single Search**: `uv run python -m src search "What is PagedAttention?" --k 10`
- **Batch Dataset Search**: `uv run python -m src search_dataset --dataset_path <path> --k 10 --save_directory <out_dir>`
- **Single Answer**: `uv run python -m src answer "How to configure PagedAttention?" --k 10`
- **Batch Answer Generation**: `uv run python -m src answer_dataset --student_search_results_path <path> --save_directory <out_dir>`
- **Recall Evaluation**: `uv run python -m src evaluate --student_search_results_path <results.json> --dataset_path <ground_truth.json>`
- **Launch REST Server**: `uv run python -m src serve --host 0.0.0.0 --port 8000`

---

## 10. Performance Benchmarks & Key Design Decisions

### 10.1 Evaluation Scorecard

| Evaluation Dataset | Metric | System Result | 42 Project Requirement | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Documentation Dataset** | Recall@5 | **81.0%** | $\ge 80.0\%$ | ✅ PASSED |
| **Documentation Dataset** | Recall@10 | **89.0%** | — | ✅ HIGH ACCURACY |
| **Codebase Dataset** | Recall@5 | **55.6%** | $\ge 50.0\%$ | ✅ PASSED |
| **Codebase Dataset** | Recall@10 | **59.6%** | — | ✅ HIGH ACCURACY |
| **Full Indexing Speed** | Ingestion & Storage | **< 3 minutes** | $\le 5$ minutes | ✅ FAST |
| **Batch Retrieval Speed** | 200 Questions Throughput | **< 45 seconds** | $\le 90$ seconds | ✅ FAST |

### 10.2 Architectural Trade-Offs & Decisions

```
+-----------------------------------------------------------------------------------------+
|                                KEY DESIGN DECISIONS                                     |
+-----------------------------------------------------------------------------------------+
| 1. Dual Indexing (BM25 + Dense Embeddings)                                             |
|    - BM25 excels at exact symbol names (e.g. `LLMEngine`, `--tensor-parallel-size`).    |
|    - MiniLM embeddings capture conceptual questions without keyword overlap.            |
|    - Solution: Combine via WSF with 0.8 lexical / 0.2 semantic weighting.              |
+-----------------------------------------------------------------------------------------+
| 2. Incremental Change Tracking via manifest.json                                       |
|    - Scanning thousands of files repeatedly is slow.                                   |
|    - Solution: Compare mtime first (O(1)); if changed, compare SHA-256 hash.           |
|    - Only modified files re-trigger embedding and ChromaDB upsert operations.          |
+-----------------------------------------------------------------------------------------+
| 3. Invariant-Checked Character Slicing                                                  |
|    - Splitters can strip whitespace and corrupt character offsets.                     |
|    - Solution: Validate `content[first:last] == chunk.text`; repair via `find()` if     |
|      needed. Guarantees 100% compliance with evaluation IoU calculations.               |
+-----------------------------------------------------------------------------------------+
| 4. Deterministic Local Model (Qwen3-0.6B)                                               |
|    - Completely offline; no external API fees, rate limits, or network dependencies.   |
|    - `temperature=0.0` and `do_sample=False` guarantee reproducible outputs.            |
+-----------------------------------------------------------------------------------------+
```
