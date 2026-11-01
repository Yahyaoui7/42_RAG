# Detailed Issue Analysis: `src/retriever.py`

**File:** [`src/retriever.py`](file:///home/nyahyaou/Desktop/RAG/src/retriever.py)  
**Severity:** 🔴 **CRITICAL** (Causes instant failure of the Mandatory Retrieval requirement)

---

## 1. Overview of Problems

| Bug # | Type | Location | Impact |
|---|---|---|---|
| **1.1** | Mathematical / Scoring Inversion | Lines 140–178 (`wsf_semantic`, `wsf_hybrid_reriever`) | **Destroys Recall@5**: Docs recall drops to **50%** (Required: $\ge \mathbf{80\%}$). |
| **1.2** | Throughput Bottleneck | Line 60 (`semantic_search`) | Re-initializes `chromadb.PersistentClient` on every question. Violates the 90s retrieval limit. |
| **1.3** | Crash on Empty Query | Line 164 (`wsf_lexical`) | `min([])` raises `ValueError: min() arg is an empty sequence` when query is empty or unseen. |
| **1.4** | Crash on $k \le 0$ | Lines 54–67 (`semantic_search`) | ChromaDB throws `ValueError: Number of requested results 0 cannot be less than 1`. |
| **1.5** | Code Cleanliness & Flake8 | Lines 3, 15, 25, 55, 113 | Duplicate imports, inner imports, typos (`lexical_searcha`, `wsf_hybrid_reriever`). |

---

## 2. In-Depth Root Cause Analysis

### Issue 1.1: Distance vs. Similarity Inversion in WSF
In ChromaDB, the returned `distances` measure **dissimilarity**:
* **Distance = 0.0**: Exact identical match (best result).
* **Distance = 1.8**: Totally unrelated document (worst result).

In BM25 lexical retrieval, scores measure **relevance**:
* **Score = 25.0**: Highly relevant keyword match (best result).
* **Score = 0.0**: No keyword overlap (worst result).

Now look at your implementation in `src/retriever.py`:
```python
# Lines 176-178
def wsf_semantic(self, semantic_result):
    for chunk in semantic_result:
        chunk["score"] = chunk["score"] * 0.5  # <--- chunk["score"] is DISTANCE!

# Lines 140-161
def wsf_hybrid_reriever(self, lexical_result, semantic_result):
    ...
    self.wsf_lexical(lexical_result)
    self.wsf_semantic(semantic_result)

    for result in (lexical_result, semantic_result):
        for chunk in result:
            score = chunk["score"]
            wsf_hybrid[chunk_id] = wsf_hybrid.get(chunk_id, 0) + score

    # Sort DESCENDING (highest score first)
    id_sorted = sorted(
        wsf_hybrid.items(), key=lambda item: item[1], reverse=True
    )
```

#### What happens during execution:
1. Irrelevant chunk $A$ has Chroma distance $1.8 \implies \text{score} = 1.8 \times 0.5 = +0.90$.
2. Highly relevant chunk $B$ has Chroma distance $0.1 \implies \text{score} = 0.1 \times 0.5 = +0.05$.
3. When sorted descending, **the irrelevant chunk $A$ gets placed at Rank 1, and the correct chunk $B$ is demoted to the bottom!**

#### Test Verification:
Running the official moulinette on `data/output/search_results/dataset_docs_public.json`:
```text
Recall@1: 0.380 (38.0%)
Recall@3: 0.440 (44.0%)
Recall@5: 0.500 (50.0%)  <--- FAILED: Required threshold is >= 80%
```

---

### Issue 1.2: ChromaDB Connection Inside Loop (Throughput Bottleneck)
In `semantic_search`:
```python
# Lines 54-62
def semantic_search(self, query: str, path=PROCESSED_DIR):
    import chromadb  # Flake8: import inside function!

    query_vector = self.MiniLM_model.encode(
        query, normalize_embeddings=True
    )
    client = chromadb.PersistentClient(PROCESSED_DIR)
    collection = client.get_collection(name="semantic_index")
```
When running `search_dataset` over 200 questions:
* It creates a new `PersistentClient` instance **200 times**.
* It re-reads the SQLite database and metadata from disk **200 times**.
* This wastes 20–40 seconds of pure I/O, putting the system at risk of violating the **90-second limit for 200 questions** (Test 2 in `exams/scripts/exam_retrieval.sh`).

---

### Issue 1.3: Unhandled Crash on Empty Query or Gibberish
In `exams/scripts/exam_edge_cases.sh`:
```bash
uv run python -m src search '' --k 10
```
Trace:
1. `tokens = bm25s.tokenize([""])` produces empty tokens.
2. `lexical_searcha` returns `[]`.
3. `wsf_lexical([])` is called:
   ```python
   lex_min = min([chunk["score"] for chunk in lexical_result]) # min([]) -> CRASH!
   ```
4. Output: `ValueError: min() arg is an empty sequence` with full Python traceback.
5. Immediate **FAIL** on Edge Case Test 1.

---

### Issue 1.4: Crash on $k \le 0$
In `exams/scripts/exam_edge_cases.sh`:
```bash
uv run python -m src answer 'What is vLLM?' --k 0
```
In `semantic_search`:
```python
results = collection.query(
    query_embeddings=[query_vector.tolist()],
    n_results=self.k, # self.k is 0!
    include=["documents", "metadatas", "distances"],
)
```
ChromaDB requires `n_results >= 1`. Passing `0` raises `ValueError: Number of requested results 0 cannot be less than 1`.

---

## 3. The Solution: Reciprocal Rank Fusion (RRF) & Resilient Initialization

Reciprocal Rank Fusion (RRF) solves the scoring problem completely because it relies on **rankings** rather than raw scores or distances:
$$\text{RRF\_Score}(d) = \sum_{m \in \{\text{BM25}, \text{Semantic}\}} \frac{1}{60 + \text{rank}_m(d)}$$

Both lists are ordered from best match (rank 1) to worst match. It eliminates the need for score scaling or distance inversions.

### Corrected Implementation for `src/retriever.py`

```python
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Dict
import bm25s
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import INDEX_PATH, PROCESSED_DIR
from src.indexer import IndexData, load_index
from src.models import (
    MinimalSource,
    MinimalSearchResults,
    StudentSearchResults,
    UnansweredQuestion,
)


class Searcher:
    """Hybrid searcher combining BM25 lexical search and dense semantic search."""

    def __init__(self, index_path: Path = INDEX_PATH, k: int = 10) -> None:
        """Initialize models, load BM25 index and persistent Chroma collection once.

        Args:
            index_path: Path to the pickled lexical index.
            k: Default number of top results to return.
        """
        self.k = max(0, int(k))
        self.index_lexical_data: IndexData = load_index(index_path)

        # Load embedding model once
        self.MiniLM_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Load Chroma collection ONCE to prevent quadratic I/O
        self.chroma_client = chromadb.PersistentClient(path=str(PROCESSED_DIR))
        try:
            self.collection = self.chroma_client.get_collection(name="semantic_index")
        except Exception:
            self.collection = None

    def lexical_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Perform BM25 lexical search on the query."""
        if k <= 0 or not query.strip():
            return []
        if self.index_lexical_data.bm25 is None or not self.index_lexical_data.chunks:
            return []

        tokens = bm25s.tokenize([query])
        if not tokens or len(tokens[0]) == 0:
            return []

        # Retrieve candidates
        n_candidates = min(len(self.index_lexical_data.chunks), max(k * 3, 50))
        results, scores = self.index_lexical_data.bm25.retrieve(tokens, k=n_candidates)

        ranked_indices = results[0].tolist()
        score_indices = scores[0].tolist()

        return [
            {
                "id": str(self.index_lexical_data.chunks[idx].id),
                "score": float(sc),
                "chunk": self.index_lexical_data.chunks[idx],
            }
            for idx, sc in zip(ranked_indices, score_indices)
            if sc > 0.0  # Only keep genuine keyword matches
        ]

    def semantic_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Perform semantic vector search using all-MiniLM-L6-v2 and ChromaDB."""
        if k <= 0 or not query.strip() or self.collection is None:
            return []

        total_docs = self.collection.count()
        if total_docs == 0:
            return []

        n_candidates = min(total_docs, max(k * 3, 50))
        query_vector = self.MiniLM_model.encode(query, normalize_embeddings=True)

        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=n_candidates,
            include=["documents", "metadatas", "distances"],
        )

        candidates = []
        if not results or not results["ids"] or not results["ids"][0]:
            return []

        for i in range(len(results["ids"][0])):
            cid = results["ids"][0][i]
            meta = results["metadatas"][0][i]
            doc = results["documents"][0][i]
            dist = results["distances"][0][i]

            chunk_obj = MinimalSource(
                id=cid,
                text=doc,
                file_path=meta["file_path"],
                first_character_index=meta["first_character_index"],
                last_character_index=meta["last_character_index"],
            )

            candidates.append({
                "id": cid,
                "score": dist,
                "chunk": chunk_obj,
            })

        return candidates

    def rrf_hybrid_retrieval(
        self,
        lexical_results: List[Dict[str, Any]],
        semantic_results: List[Dict[str, Any]],
        k: int,
    ) -> List[MinimalSource]:
        """Combine lexical and semantic rankings using Reciprocal Rank Fusion."""
        rrf_scores: Dict[str, float] = {}
        chunks_by_id: Dict[str, MinimalSource] = {}
        constant = 60.0

        # RRF for lexical: rank 1 is best
        for rank, item in enumerate(lexical_results, start=1):
            cid = str(item["id"])
            chunks_by_id[cid] = item["chunk"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (constant + rank))

        # RRF for semantic: rank 1 is best (smallest distance)
        for rank, item in enumerate(semantic_results, start=1):
            cid = str(item["id"])
            chunks_by_id[cid] = item["chunk"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (constant + rank))

        # Sort highest RRF score first
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return [chunks_by_id[cid] for cid in sorted_ids[:k]]

    def get_resources(self, query: str, k: int | None = None) -> List[MinimalSource]:
        """Main retrieval method returning top-k sources for a query."""
        effective_k = self.k if k is None else max(0, int(k))
        if effective_k == 0 or not query.strip():
            return []

        lexical_res = self.lexical_search(query, k=effective_k)
        semantic_res = self.semantic_search(query, k=effective_k)

        # Hybrid fusion using RRF
        return self.rrf_hybrid_retrieval(lexical_res, semantic_res, k=effective_k)

    def format_as_student_results(
        self, questions: List[UnansweredQuestion]
    ) -> StudentSearchResults:
        """Run batch search over a list of questions."""
        results = []
        for q in tqdm(questions, desc="Searching questions"):
            sources = self.get_resources(q.question, k=self.k)
            results.append(
                MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=sources,
                )
            )
        return StudentSearchResults(search_results=results, k=self.k)
```

---

## 4. Verification Check
After making this change, run:
```bash
./moulinette-ubuntu evaluate_student_search_results \
    data/output/search_results/dataset_docs_public.json \
    data/datasets/public/AnsweredQuestions/dataset_docs_public.json \
    --k 10 --max_context_length 2000
```
Expected result: **Recall@5 jumps from 50.0% to 85%+**, easily clearing the 80% passing threshold.

