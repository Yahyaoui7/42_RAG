# Detailed Issue Analysis: `src/__main__.py`

**File:** [`src/__main__.py`](file:///home/nyahyaou/Desktop/RAG/src/__main__.py)  
**Severity:** 🔴 **CRITICAL** (Fails Edge Case Tests in `exam_edge_cases.sh` and violates CLI specifications)

---

## 1. Overview of Problems

| Bug # | Type | Location | Impact |
|---|---|---|---|
| **2.1** | Missing Argument Default & Unused Flag | Line 62 (`answer`) | `answer(self, query, k)` has no default for `k`. `k` is not passed to `Searcher()`. |
| **2.2** | Unhandled Traceback on Missing File | Lines 48–50 (`search_dataset`) | `Path(dataset_path).read_text()` crashes with `FileNotFoundError` (fails Edge Case Test 4). |
| **2.3** | Output String Format Mismatch | Line 39 (`search`) | Prints `[0: 1352]` with space after colon. Subject specifies `[0:1352]`. |
| **2.4** | Missing Error Handling on Bad Inputs | Entire CLI class | Missing `try...except` guards around file I/O, malformed JSON, and empty arguments. |
| **2.5** | Flake8 & PEP 8 Violations | Lines 2, 16, 18, 117–126 | `import fire, sys` (E401), unused imports (`uuid`, `json`), commented-out code at bottom. |
| **2.6** | Mypy / Typing Violations | Lines 42, 62, 79, 94 | Missing parameter types and return type annotations (`-> None`). |
| **2.7** | Code Quality / Typos | Lines 88, 101 | `genreter = Generate()`, `evaluater = Evaluator()`. |

---

## 2. In-Depth Root Cause Analysis

### Issue 2.1: Broken `answer` Command
Look at lines 62–74 of `src/__main__.py`:
```python
def answer(self, query, k):  # <--- No default value for k!
    cache = Caching(k)
    cache_answer = cache.get_answer_from_cache(query)
    if cache_answer:
        print("answer: ", cache_answer)
    else:
        searcher = Searcher()  # <--- k is NEVER passed! Uses Searcher default k=10
        model = Generate()

        sources = searcher.get_resources(query)  # <--- k is NOT passed to get_resources!
        prompt = Prompt().build_prompt(query, sources)
        answer = model.generate_answer(prompt)
```

1. If the user invokes `uv run python -m src answer "What is vLLM?"` without `--k`, Python Fire crashes because `k` is a required positional argument!
2. If the user passes `--k 3`, `Searcher()` still uses default `k=10` because `k` is not forwarded to `Searcher(k=k)` or `searcher.get_resources(query, k=k)`.
3. If the user passes `--k 0`, the system crashes because `Searcher` does not handle $k \le 0$.

---

### Issue 2.2: Edge Case Test 4 Crash (Missing Dataset File)
In `exams/scripts/exam_edge_cases.sh`:
```bash
uv run python -m src search_dataset --dataset_path /nonexistent/dataset.json --k 10
```
In `src/__main__.py`:
```python
def search_dataset(self, dataset_path=SEARCH_DATASET_PATH, ...):
    questions = RagDataset.model_validate_json(
        Path(dataset_path).read_text()  # <--- CRASH! FileNotFoundError
    ).rag_questions
```
Because there is no check or `try...except`, Python prints a raw `Traceback (most recent call last)`.
`exam_edge_cases.sh` greps for `"Traceback (most recent call last)"` in `stderr` and marks **TEST 4: FAIL**.

The subject explicitly states (Chapter VI.6):
> *"The CLI is tested with such edge cases and must never crash with an unhandled traceback."*

---

### Issue 2.3: Search Output Spacing Violation
Subject Chapter VI.3 (page 13) shows the exact expected terminal output:
```text
wil@42:~/rag$ uv run python -m src search "How to configure the OpenAI server?" --k 5
data/raw/vllm-0.10.1/examples/online_serving/openai_transcription_client.py [0:1352]
data/raw/vllm-0.10.1/examples/online_serving/prompt_embed_inference_with_openai_client.py [0:932]
```
In your code (line 39):
```python
print(f"{chunk.file_path} [{chunk.first_character_index}: {chunk.last_character_index}]")
```
Notice the space: `[{chunk.first_character_index}: {chunk.last_character_index}]` produces:
`data/raw/vllm-0.10.1/... [0: 1352]`
Correction: Remove the space after `:`: `f"{chunk.file_path} [{chunk.first_character_index}:{chunk.last_character_index}]"`.

---

### Issue 2.4: Flake8 & Mypy Issues
* Line 2: `import fire, sys` violates Flake8 `E401` (multiple imports on one line).
* Lines 16 & 18: `import uuid` and `import json` are imported but never used (`F401`).
* Lines 117–126: Dead commented-out code at the bottom of the file.
* Lines 42, 62, 79, 94: None of these methods have type annotations. Under `mypy --disallow-untyped-defs`, static analysis fails completely.

---

## 3. Corrected Implementation for `src/__main__.py`

```python
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
import fire

from src.caching import Caching
from src.config import (
    ANSWER_SAVE_DIRECTORY,
    ANSWERED_DATASET_PATH,
    MAX_CHUNK_SIZE,
    SEARCH_DATASET_PATH,
    SEARCH_SAVE_DIRECTORY,
    STUDENT_SEARCH_RESULTS_PATH,
    TOP_K,
)
from src.evaluator import Evaluator
from src.generate import Generate
from src.indexer import build_and_save_from_raw
from src.models import RagDataset, StudentSearchResults
from src.prompts import Prompt
from src.retriever import Searcher


class CLI:
    """Command-line interface for the RAG system built with Python Fire."""

    def index(self, max_chunk_size: int = MAX_CHUNK_SIZE) -> None:
        """Ingest data/raw/ and build indices under data/processed/.

        Args:
            max_chunk_size: Maximum character length per chunk (<= 2000).
        """
        try:
            max_size = int(max_chunk_size)
            if max_size <= 0:
                print("Error: max_chunk_size must be positive.", file=sys.stderr)
                return
            build_and_save_from_raw(max_size)
        except Exception as e:
            print(f"Error indexing corpus: {e}", file=sys.stderr)

    def search(self, query: str = "", k: int = 10) -> None:
        """Return top-k sources for a single query.

        Args:
            query: Question or keywords to search.
            k: Number of source locations to retrieve.
        """
        try:
            k_val = max(0, int(k))
            if not query or not query.strip() or k_val == 0:
                return

            searcher = Searcher(k=k_val)
            results = searcher.get_resources(query, k=k_val)

            for chunk in results:
                # Exact format required: [start:end] without space
                print(f"{chunk.file_path} [{chunk.first_character_index}:{chunk.last_character_index}]")
        except Exception as e:
            print(f"Error during search: {e}", file=sys.stderr)

    def search_dataset(
        self,
        dataset_path: str = SEARCH_DATASET_PATH,
        save_directory: str = SEARCH_SAVE_DIRECTORY,
        k: int = TOP_K,
    ) -> None:
        """Run search across an entire question dataset and output JSON results.

        Args:
            dataset_path: Path to dataset JSON file.
            save_directory: Directory where student search results will be saved.
            k: Number of results to retrieve per question.
        """
        try:
            path = Path(dataset_path)
            if not path.is_file():
                print(f"Error: Dataset file not found at {dataset_path}", file=sys.stderr)
                return

            k_val = max(0, int(k))
            raw_content = path.read_text(encoding="utf-8")
            questions = RagDataset.model_validate_json(raw_content).rag_questions

            searcher = Searcher(k=k_val)
            result = searcher.format_as_student_results(questions)

            out_dir = Path(save_directory)
            out_dir.mkdir(parents=True, exist_ok=True)
            output_file = out_dir / path.name
            output_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            print(f"Saved student_search_results to {output_file}")
        except Exception as e:
            print(f"Error processing search_dataset: {e}", file=sys.stderr)

    def answer(self, query: str = "", k: int = 10) -> None:
        """Answer a single query using retrieved context.

        Args:
            query: Question text.
            k: Number of sources to retrieve for augmentation.
        """
        try:
            k_val = max(0, int(k))
            if not query or not query.strip() or k_val == 0:
                print("answer: I don't know.")
                return

            cache = Caching(top_k=k_val)
            cached = cache.get_answer_from_cache(query)
            if cached:
                print("answer:", cached)
                return

            searcher = Searcher(k=k_val)
            sources = searcher.get_resources(query, k=k_val)
            prompt = Prompt().build_prompt(query, sources)

            generator = Generate()
            answer_text = generator.generate_answer(prompt)

            cache.save_query_and_answer(query, answer_text)
            print("answer:", answer_text)
        except Exception as e:
            print(f"Error generating answer: {e}", file=sys.stderr)

    def answer_dataset(
        self,
        student_search_results_path: str = STUDENT_SEARCH_RESULTS_PATH,
        save_directory: str = ANSWER_SAVE_DIRECTORY,
    ) -> None:
        """Generate answers for all questions in a student search results file.

        Args:
            student_search_results_path: Path to the search results JSON.
            save_directory: Output directory for the answered JSON file.
        """
        try:
            input_path = Path(student_search_results_path)
            if not input_path.is_file():
                print(f"Error: Search results file not found at {input_path}", file=sys.stderr)
                return

            content = input_path.read_text(encoding="utf-8")
            student_result = StudentSearchResults.model_validate_json(content)

            out_dir = Path(save_directory)
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / input_path.name

            generator = Generate()
            generator.answer_dataset(student_result, output_path)
            print(f"Saved student_search_results_and_answer to {output_path}")
        except Exception as e:
            print(f"Error running answer_dataset: {e}", file=sys.stderr)

    def evaluate(
        self,
        student_search_results_path: str = STUDENT_SEARCH_RESULTS_PATH,
        dataset_path: str = ANSWERED_DATASET_PATH,
    ) -> None:
        """Evaluate Recall@k scores against ground truth.

        Args:
            student_search_results_path: Path to student search results JSON.
            dataset_path: Path to ground-truth AnsweredQuestions JSON.
        """
        try:
            if not Path(student_search_results_path).is_file():
                print(f"Error: Missing search results at {student_search_results_path}", file=sys.stderr)
                return
            if not Path(dataset_path).is_file():
                print(f"Error: Missing ground truth dataset at {dataset_path}", file=sys.stderr)
                return

            evaluator = Evaluator(student_search_results_path, dataset_path)
            scores = evaluator.get_score()

            print("\nEvaluation Results")
            print("========================================")
            for k_val, val in scores.items():
                print(f"Recall@{k_val}: {val:.3f} ({val * 100:.1f}%)")
        except Exception as e:
            print(f"Error during evaluation: {e}", file=sys.stderr)


def main() -> None:
    """CLI entry point using Python Fire."""
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
```

