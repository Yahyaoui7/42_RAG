# Master Audit & Issues Index

This directory contains comprehensive, file-by-file issue reports for the **RAG Against the Machine** project (42 Curriculum, Version 2.0).

Each document details:
1. Exact file location and line numbers.
2. The specific project/subject requirements.
3. The root cause of the bug or shortcoming.
4. Problematic code vs. corrected code.
5. Edge cases and testing commands.

---

## Issue Files Summary

| File | Associated Component | Severity | Description |
|---|---|---|---|
| [`retriever_issues.md`](retriever_issues.md) | `src/retriever.py` | 🔴 **CRITICAL** | Inverted semantic distance destroys Recall@5 (50% vs 80% req); ChromaDB query throughput bottleneck; empty query and $k=0$ crashes. |
| [`main_issues.md`](main_issues.md) | `src/__main__.py` | 🔴 **CRITICAL** | CLI edge cases fail `exam_edge_cases.sh` (tracebacks on empty query, $k=0$, bad path); `answer` ignores $k$; colon format spacing. |
| [`models_issues.md`](models_issues.md) | `src/models.py` | 🟡 **HIGH** | `MinimalSource` serializes internal `id` and full `text` to search results, causing 10x file bloating and schema mismatch in moulinette. |
| [`indexer_issues.md`](indexer_issues.md) | `src/indexer.py` | 🟡 **HIGH** | Quadratic I/O bottleneck in `save_metadata_file` (rewrites JSON 2,000 times); ChromaDB stale chunks bug on incremental indexing (Bonus 3). |
| [`evaluator_issues.md`](evaluator_issues.md) | `src/evaluator.py` | 🟡 **HIGH** | `KeyError` / `IndexError` when dataset contains unanswered questions; unhandled file read tracebacks. |
| [`caching_issues.md`](caching_issues.md) | `src/caching.py` | 🔵 **BONUS** | Bonus 4 requires caching *both* index and query results; search queries not cached; directory creation missing (`FileNotFoundError`). |
| [`chunking_issues.md`](chunking_issues.md) | `src/chunking/` | 🟢 **MEDIUM** | Unremoved debug prints; `validate_chunk` not enforced; character range edge cases. |
| [`makefile_issues.md`](makefile_issues.md) | `Makefile` | 🟢 **MEDIUM** | `clean` relies on undeclared `pyclean`; `debug` has invalid pdb syntax; hardcoded evaluation paths. |
| [`readme_issues.md`](readme_issues.md) | `README.md` | 🔴 **CRITICAL** | Unfinished "TODO" sections; missing all mandatory Chapter VIII sections (System Architecture, AI Usage, Chunking Strategy, etc.). |
| [`bonus_local_http_api.md`](bonus_local_http_api.md) | *Bonus 5 (Missing)* | 🔵 **BONUS** | Missing Local HTTP API implementation guide and full code for FastAPI / HTTP server endpoints. |

---

## Evaluation Failure Points Summary

If submitted in its current state:
1. **Recall@5 Test (`exam_retrieval.sh`)**: **FAILS**. Docs recall is currently **50.0%** (minimum threshold is **80.0%**).
2. **Edge Cases Test (`exam_edge_cases.sh`)**: **FAILS**. Crashes with unhandled Python tracebacks on empty query, $k=0$, and missing dataset file.
3. **Documentation Review**: **FAILS**. `README.md` contains "TODO" and links to non-existent `docs/`.
4. **Code Quality Review (`make lint`)**: **FAILS**. Multiple flake8 syntax violations, missing type hints on most functions, and missing PEP 257 docstrings.

