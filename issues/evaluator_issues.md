# Detailed Issue Analysis: `src/evaluator.py`

**File:** [`src/evaluator.py`](file:///home/nyahyaou/Desktop/RAG/src/evaluator.py)  
**Severity:** 🟡 **HIGH** (Unhandled exceptions, `KeyError` crashes, missing types)

---

## 1. Overview of Problems

| Bug # | Type | Location | Impact |
|---|---|---|---|
| **5.1** | `KeyError: 'sources'` on Unanswered Datasets | Line 79 | If `evaluate` is run against an unanswered dataset or malformed ground truth, it crashes with `KeyError`. |
| **5.2** | `IndexError: list index out of range` | Line 79 | Assumes every question has at least one source (`[0]`). Crashes if `sources` is empty. |
| **5.3** | `KeyError` on Missing Question ID | Line 83 | If a question ID in the dataset is not present in the student results, indexing `self.result_dict[qid]` raises `KeyError`. |
| **5.4** | Division by Zero Risk | Lines 48–61 (`check_is_valid`) | If both spans are zero length, `all_range` is 0 $\rightarrow$ `ZeroDivisionError`. |
| **5.5** | Flake8 & Cleanliness | Lines 9, 19–36 | Typo in comment (`teh`), commented-out dead code block, missing type annotations. |

---

## 2. In-Depth Root Cause Analysis

### Issue 5.1 & 5.2: Assumptions About `question["sources"][0]`
Look at lines 76–85 of `src/evaluator.py`:
```python
for question in questions:
    raw_source = MinimalSource.model_validate(
        question["sources"][0]  # <--- Crashes if "sources" missing or empty!
    )
    student_source = [
        MinimalSource.model_validate(source)
        for source in self.result_dict[question["question_id"]]  # <--- Crashes if question_id missing!
    ]
```

1. **Missing `"sources"` Key**:
   The subject defines `RagDataset` as:
   ```python
   class RagDataset(BaseModel):
       rag_questions: List[AnsweredQuestion | UnansweredQuestion]
   ```
   If a user mistakenly passes an unanswered dataset (`dataset_docs_public.json`), `question` has no `"sources"` key $\rightarrow$ unhandled `KeyError: 'sources'`.
2. **Multiple Ground-Truth Sources**:
   Many questions in the benchmark have **more than one** ground-truth source (`sources: [source1, source2]`). Your code only checks `sources[0]`, ignoring any secondary valid sources!
3. **Missing Question ID in Student Output**:
   If the student results file is incomplete or missing a question, accessing `self.result_dict[question["question_id"]]` throws `KeyError`.

---

## 3. Corrected Implementation for `src/evaluator.py`

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from src.models import MinimalSource


class Evaluator:
    """Evaluates student retrieval Recall@k against ground-truth datasets."""

    def __init__(self, student_search_results_path: str, dataset_path: str) -> None:
        """Load student results and ground truth dataset safely."""
        self.student_result: Dict[str, Any] = json.loads(
            Path(student_search_results_path).read_text(encoding="utf-8")
        )
        self.dataset_result: Dict[str, Any] = json.loads(
            Path(dataset_path).read_text(encoding="utf-8")
        )

        self.result_dict: Dict[str, List[Dict[str, Any]]] = {
            item["question_id"]: item.get("retrieved_sources", [])
            for item in self.student_result.get("search_results", [])
        }

    def check_is_valid(self, raw_source: MinimalSource, student_source: MinimalSource) -> bool:
        """Check if student source overlaps with raw source in the same file (IoU >= 0.05)."""
        if raw_source.file_path != student_source.file_path:
            return False

        first_inter = max(raw_source.first_character_index, student_source.first_character_index)
        last_inter = min(raw_source.last_character_index, student_source.last_character_index)
        intersection = max(0, last_inter - first_inter)

        first_union = min(raw_source.first_character_index, student_source.first_character_index)
        last_union = max(raw_source.last_character_index, student_source.last_character_index)
        union = max(1, last_union - first_union)

        return (intersection / union) >= 0.05

    def check_top_k(
        self,
        raw_sources: List[MinimalSource],
        student_sources: List[MinimalSource],
        k: int,
    ) -> bool:
        """Return True if at least one raw source is retrieved within the student top-k."""
        for cand in student_sources[:k]:
            for raw in raw_sources:
                if self.check_is_valid(raw, cand):
                    return True
        return False

    def get_score(self) -> Dict[int, float]:
        """Compute Recall@k for k in [1, 3, 5, 10]."""
        questions = self.dataset_result.get("rag_questions", [])
        valid_eval_questions = [q for q in questions if "sources" in q and len(q["sources"]) > 0]

        if not valid_eval_questions:
            return {1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0}

        checkpoints = [1, 3, 5, 10]
        score_dict: Dict[int, float] = {}

        for k in checkpoints:
            hits = 0
            for q in valid_eval_questions:
                raw_sources = [MinimalSource.model_validate(s) for s in q["sources"]]
                retrieved_raw = self.result_dict.get(q["question_id"], [])
                student_sources = [MinimalSource.model_validate(s) for s in retrieved_raw]

                if self.check_top_k(raw_sources, student_sources, k):
                    hits += 1

            score_dict[k] = hits / len(valid_eval_questions)

        return score_dict
```

