# Detailed Issue Analysis: `src/models.py`

**File:** [`src/models.py`](file:///home/nyahyaou/Desktop/RAG/src/models.py)  
**Severity:** 🟡 **HIGH** (Schema bloating, serialization mismatch with exam scripts)

---

## 1. Overview of Problems

| Bug # | Type | Location | Impact |
|---|---|---|---|
| **4.1** | Schema Bloating / Non-conformance | Lines 6–11 (`MinimalSource`) | `id` and `text` are serialized into `search_results.json`. Causes 10x file bloating and schema errors in moulinette. |
| **4.2** | Missing Future Annotations | Line 25 (`RagDataset`) | Using `AnsweredQuestion | UnansweredQuestion` without `from __future__ import annotations` can trigger runtime `TypeError` in Python 3.10. |
| **4.3** | Missing PEP 257 Docstrings | Entire file | Pydantic models lack docstrings describing fields and validation rules. |

---

## 2. In-Depth Root Cause Analysis

### Issue 4.1: Extra Fields in `MinimalSource` Serialized to Disk
Look at lines 6–11 of `src/models.py`:
```python
class MinimalSource(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    text: str = Field(default="")
    file_path: str
    first_character_index: int
    last_character_index: int
```

When `search_dataset` serializes `StudentSearchResults` to disk:
```json
{
  "file_path": "data/raw/vllm-0.10.1/docs/features/lora.md",
  "first_character_index": 4695,
  "last_character_index": 6098,
  "id": "f99d3bcc-e2b6-4216-83f6-a066b27c5db1",
  "text": "### Using API Endpoints\n\nLoading a LoRA Adapter:\n\nTo dynamically load..."
}
```

Now compare with Chapter VI.5 & VI.6 of the subject (pages 14–16):
```text
Source information: Each MinimalSource contains:
• file_path: path to the source file, relative to your project root...
• first_character_index: Starting character position
• last_character_index: Ending character position
```

And the exact example given on page 16:
```json
"retrieved_sources": [
  {
    "file_path": "data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md",
    "first_character_index": 9867,
    "last_character_index": 10100
  }
]
```

#### Why this is problematic:
1. **File Bloating**: `dataset_docs_public.json` grows from ~120 KB to **1.6 MB**.
2. **Schema Validation Warnings**: In `exams/scripts/exam_answer.sh` line 141:
   ```text
   WARNING: list_valid_questions failed. This may indicate a schema mismatch in the search results file.
   ```
   The evaluator script flags extra fields as a potential schema violation.

#### The Solution:
You need `id` and `text` internally for ChromaDB and prompting. In Pydantic, you can mark fields with `exclude=True`:
```python
class MinimalSource(BaseModel):
    id: UUID = Field(default_factory=uuid4, exclude=True)
    text: str = Field(default="", exclude=True)
    file_path: str
    first_character_index: int
    last_character_index: int
```
When `model.model_dump_json()` or `model.model_dump()` is called, `id` and `text` are **automatically omitted from the JSON output**, producing clean, compliant files that match the subject verbatim!

---

## 3. Corrected Implementation for `src/models.py`

```python
from __future__ import annotations

from typing import List, Union
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Represents a character span in a source code or documentation file."""
    id: UUID = Field(default_factory=uuid4, exclude=True)
    text: str = Field(default="", exclude=True)
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Represents a question waiting for retrieval and answer."""
    question_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represents a question with ground-truth sources and reference answer."""
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Container for a list of evaluation questions."""
    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]


class MinimalSearchResults(BaseModel):
    """Search results for a single question."""
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Answer and retrieved sources for a single question."""
    answer: str


class StudentSearchResults(BaseModel):
    """Collection of search results across multiple questions."""
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Collection of search results with generated answers across multiple questions."""
    search_results: List[MinimalAnswer]
    k: int
```

