# Detailed Issue Analysis: `src/chunking/`

**Files:**
- [`src/chunking/text_chunker.py`](file:///home/nyahyaou/Desktop/RAG/src/chunking/text_chunker.py)
- [`src/chunking/base.py`](file:///home/nyahyaou/Desktop/RAG/src/chunking/base.py)
- [`src/chunking/python_chunker.py`](file:///home/nyahyaou/Desktop/RAG/src/chunking/python_chunker.py)
- [`src/chunking/markdown_chunker.py`](file:///home/nyahyaou/Desktop/RAG/src/chunking/markdown_chunker.py)

**Severity:** 🟢 **MEDIUM** (Hard limit violation risks, debug print statements, uncalled validation)

---

## 1. Overview of Problems

| Bug # | Type | Location | Impact |
|---|---|---|---|
| **7.1** | Unremoved Debug Print | `text_chunker.py` line 41 | Prints `"the content[start:end] != text"` to standard output during indexing. |
| **7.2** | Missing Invariant Validation | `base.py` & `indexer.py` | `validate_chunk()` is implemented in `base.py` but **never called**. An oversized chunk (> 2000 chars) invalidates the entire submission. |
| **7.3** | Fallback Index Alignment Failure | All chunkers lines 41–45 | If `content.find(text, start)` returns `-1`, `end` calculation remains mismatched with original text. |

---

## 2. In-Depth Root Cause Analysis

### Issue 7.1: Leftover Debug Print
In [`src/chunking/text_chunker.py`](file:///home/nyahyaou/Desktop/RAG/src/chunking/text_chunker.py#L40-L45):
```python
if content[start:end] != text:
    print("the content[start:end] != text")  # <--- Leftover debugging print!
    found = content.find(text, start)
    if found != -1:
        start = found
        end = start + len(text)
```
Whenever a text file has a slight offset difference, this print fires during `uv run python -m src index`, cluttering stdout and violating clean CLI output guidelines.

---

### Issue 7.2: The 2,000 Character Hard Invariant
Subject Chapter VI.1 & VI.2 (page 11 & 12) states:
> *"Do not go above it: the moulinette rejects any retrieved source longer than 2000 characters (its max_context_length), and a single over-long source makes your whole output invalid."*

In `src/chunking/base.py`, you have a great validation function:
```python
def validate_chunk(chunk: Chunk, file_content: str, max_chunk_size: int = MAX_CHUNK_SIZE) -> bool:
```
**However, this is never called in `indexer.py`!** If LangChain emits a chunk with `len(chunk.text) > 2000` (e.g. on huge unbroken single-line strings or long docstrings), it is saved directly into the index.

#### Fix:
In all chunkers and in `indexer.py`, enforce an explicit clamp/filter:
```python
if len(text) > 2000 or (end - start) > 2000:
    end = start + min(len(text), 2000)
    text = content[start:end]
```

---

## 3. Cleaned Implementation for `src/chunking/text_chunker.py`

```python
from __future__ import annotations

from pathlib import Path
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import MAX_CHUNK_SIZE
from src.models import MinimalSource


def chunk_text_file(
    file_path: Path, max_chunk_size: int = MAX_CHUNK_SIZE
) -> List[MinimalSource]:
    """Split plain text files into bounded chunks with character offsets."""
    file_path = Path(file_path)
    file_path_str = str(file_path)

    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except (OSError, UnicodeError):
        return []

    if not content.strip():
        return []

    effective_max = min(max_chunk_size, 2000)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=effective_max,
        chunk_overlap=0,
        add_start_index=True,
    )

    documents = splitter.create_documents([content])
    chunks: List[MinimalSource] = []

    for doc in documents:
        text: str = doc.page_content
        if not text.strip():
            continue

        start: int = doc.metadata.get("start_index", 0)
        end: int = start + len(text)

        # Ensure exact match with original content
        if content[start:end] != text:
            found = content.find(text, start)
            if found != -1:
                start = found
                end = start + len(text)
            else:
                continue

        # Hard clamp: never exceed 2000 characters
        if end - start > 2000:
            end = start + 2000
            text = content[start:end]

        chunks.append(
            MinimalSource(
                text=text,
                file_path=file_path_str,
                first_character_index=start,
                last_character_index=end,
            )
        )

    return chunks
```
*(Apply the same 2000-char guarantee to `python_chunker.py` and `markdown_chunker.py`)*.

