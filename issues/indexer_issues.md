# Detailed Issue Analysis: `src/indexer.py`

**File:** [`src/indexer.py`](file:///home/nyahyaou/Desktop/RAG/src/indexer.py)  
**Severity:** 🟡 **HIGH** (Severe I/O performance bottleneck, stale chunks in Bonus 3, Flake8 & type violations)

---

## 1. Overview of Problems

| Bug # | Type | Location | Impact |
|---|---|---|---|
| **3.1** | Quadratic Disk I/O Bottleneck | Lines 157, 176–193 (`save_metadata_file`) | Rewrites multi-megabyte `manifest.json` on disk **2,000 times**, risking exceeding the 5-minute indexing limit. |
| **3.2** | Stale Chunks on Re-indexing (Bonus 3) | Lines 152–170 (`build_and_save_from_raw`) | When a file is modified, its new chunks are added to ChromaDB, but **its previous chunks are never removed**, creating duplicate vectors. |
| **3.3** | Inverted Function Semantics & Spurious Print | Lines 148–150, 195–223 (`is_file_dirty`) | Function returns data (truthy) when file is *clean*, and `False` when *dirty*. Spams stdout with `print(file_path)`. |
| **3.4** | Empty Batch Crash Risk in Embeddings | Line 168 | If no files changed, `new_chunck` is empty `[]`. Passing `[]` to `SentenceTransformer.encode()` can raise errors or warnings. |
| **3.5** | Flake8 & Mypy Violations | Lines 229–238, 66, 176, 195, 225, 241 | `import hashlib` inside function; missing type annotations on multiple helper functions. |

---

## 2. In-Depth Root Cause Analysis

### Issue 3.1: Quadratic Disk I/O in `save_metadata_file`
Look at lines 147–158:
```python
for file_path in tqdm(all_files, desc="Chunking"):
    data = is_file_dirty(file_path)
    if not data:
        file_chunks = _chunk_file(file_path, max_chunk_size)
        ...
        save_metadata_file(file_path, file_chunks)
```
And inside `save_metadata_file`:
```python
def save_metadata_file(file_path, chunks, save_path=FILE_METADATA):
    if os.path.exists(save_path):
        with open(save_path, "r") as file:
            data = json.load(file)  # <--- READ 2MB-10MB JSON
    else:
        data = {}

    data[str(file_path)] = meta_data

    with open(save_path, "w") as file:
        json.dump(data, file, indent=4)  # <--- WRITE 2MB-10MB JSON
```
#### Impact:
When indexing the vLLM repository (~2,000 files):
* On the 1st file: writes 1 entry.
* On the 500th file: reads, parses, updates, and writes a 2MB JSON file.
* On the 1,500th file: reads, parses, updates, and writes a 7MB JSON file.
* Total I/O: **Gigabytes of JSON re-serialization across thousands of disk write operations!**

Subject Chapter VI.1 specifies:
> *"Indexing the whole corpus must take at most 5 minutes."*

#### Fix:
Load the manifest dictionary **once** before the loop starts, update the in-memory dictionary during the loop, and write `manifest.json` **once** when the loop finishes!

---

### Issue 3.2: Incremental Indexing Stale Chunks (Bonus 3)
Bonus 3 requires:
> *"3. Incremental indexing: when a file changes, re-index only that file instead of rebuilding the whole index."*

Currently:
1. `remove_removed_file(all_files)` removes files that were *deleted* from the filesystem.
2. BUT when an existing file is *modified*:
   - New chunks are generated.
   - `save_semantic_index` calls `collection.add(...)`.
   - The old chunks that belonged to that file **remain inside ChromaDB forever**!
   - ChromaDB now has multiple outdated versions of chunks from the same file, corrupting ranking and precision.

#### Fix:
Before chunking and re-indexing a modified file, delete its existing records from the Chroma collection:
```python
collection.delete(where={"file_path": str(file_path)})
```

---

### Issue 3.3: Inverted Logic in `is_file_dirty`
In English and programming conventions:
* `is_dirty() == True` means "has unsaved changes / needs update".
* `is_dirty() == False` means "clean / unchanged".

In your code:
```python
def is_file_dirty(file_path, save_path=FILE_METADATA):
    if file_path not in manifest:
        return False  # <--- But a new file IS dirty!
    ...
    if metadata["hash"] != current_hash:
        return False  # <--- A changed file IS dirty!
    return metadata   # <--- An unchanged file returns metadata (truthy)!
```
Then in the loop:
```python
if not data:
    print(file_path)  # <--- Debug print spams terminal
```
This inverted logic makes debugging and maintenance confusing. It should be refactored to `get_cached_metadata()` or `is_file_changed()`.

---

## 3. Corrected Implementation for `src/indexer.py`

```python
from __future__ import annotations

import json
import os
import pickle
import sys
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import bm25s
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.chunking import (
    chunk_markdown_file,
    chunk_python_file,
    chunk_text_file,
)
from src.config import (
    FILE_METADATA,
    INDEX_PATH,
    MAX_CHUNK_SIZE,
    PROCESSED_DIR,
    RAW_DIR,
)
from src.models import MinimalSource
from src.scanner import scan_files


@dataclass
class IndexData:
    """Container for the BM25 index and chunk references."""
    bm25: Optional[bm25s.BM25]
    chunks: List[MinimalSource]
    tokenized_corpus: List[List[str]]


def build_lexical_index(chunks: List[MinimalSource]) -> IndexData:
    """Build a BM25 index over all chunks."""
    if not chunks:
        return IndexData(bm25=None, chunks=[], tokenized_corpus=[])

    tokenized_corpus = bm25s.tokenize([chunk.text for chunk in chunks])
    if not tokenized_corpus or not any(tokenized_corpus):
        return IndexData(bm25=None, chunks=chunks, tokenized_corpus=[])

    retriever = bm25s.BM25()
    retriever.index(tokenized_corpus, show_progress=False)

    return IndexData(
        bm25=retriever,
        chunks=chunks,
        tokenized_corpus=tokenized_corpus,
    )


def build_semantic_index(chunks: List[MinimalSource]) -> np.ndarray:
    """Generate dense embeddings for a list of chunks using all-MiniLM-L6-v2."""
    if not chunks:
        return np.empty((0, 384), dtype=np.float32)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(
        [chunk.text for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def save_semantic_index(
    vectors: np.ndarray,
    chunks: List[MinimalSource],
    collection: chromadb.Collection,
) -> None:
    """Add chunk embeddings and metadata to ChromaDB collection in batches."""
    if len(chunks) == 0 or len(vectors) == 0:
        return

    batch_size = 2000
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_vecs = vectors[i : i + batch_size]

        collection.add(
            ids=[str(chunk.id) for chunk in batch],
            documents=[chunk.text for chunk in batch],
            embeddings=batch_vecs.tolist(),
            metadatas=[
                {
                    "file_path": chunk.file_path,
                    "first_character_index": chunk.first_character_index,
                    "last_character_index": chunk.last_character_index,
                }
                for chunk in batch
            ],
        )


def save_lexical_index(index: IndexData, path: Path = INDEX_PATH) -> None:
    """Pickle IndexData to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(index, f, protocol=4)


def load_index(path: Path = INDEX_PATH) -> IndexData:
    """Load IndexData from pickle."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Index not found at {path}. Run `uv run python -m src index` first."
        )
    with path.open("rb") as f:
        return pickle.load(f)


def _chunk_file(path: Path, max_chunk_size: int) -> List[MinimalSource]:
    """Dispatch to proper chunker based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".py":
        return chunk_python_file(path, max_chunk_size=max_chunk_size)
    elif suffix == ".md":
        return chunk_markdown_file(path, max_chunk_size=max_chunk_size)
    else:
        return chunk_text_file(path, max_chunk_size=max_chunk_size)


def calculate_hash(file_path: Path) -> str:
    """Calculate SHA256 checksum of file content."""
    sha256 = hashlib.sha256()
    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()


def build_and_save_from_raw(max_chunk_size: int = MAX_CHUNK_SIZE) -> None:
    """Scan raw documents, update incremental indices and manifest."""
    files_dict = scan_files(RAW_DIR)
    all_files: List[Path] = []
    for lst in files_dict.values():
        all_files.extend(lst)
    all_files = sorted(all_files)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(FILE_METADATA)

    # Load existing manifest ONCE
    manifest: Dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    # Initialize Chroma client
    client = chromadb.PersistentClient(path=str(PROCESSED_DIR))
    collection = client.get_or_create_collection(name="semantic_index")

    # Step 1: Remove chunks for deleted files
    current_paths_set = {str(p) for p in all_files}
    deleted_files = set(manifest.keys()) - current_paths_set
    for del_path in deleted_files:
        try:
            collection.delete(where={"file_path": del_path})
        except Exception:
            pass
        manifest.pop(del_path, None)

    all_chunks: List[MinimalSource] = []
    new_chunks: List[MinimalSource] = []
    manifest_updated = False

    # Step 2: Process files with in-memory manifest updates
    for file_path in tqdm(all_files, desc="Chunking"):
        str_path = str(file_path)
        mtime = file_path.stat().st_mtime
        file_hash = calculate_hash(file_path)

        cached_entry = manifest.get(str_path)
        # Check if file is clean (hash matches)
        if cached_entry and cached_entry.get("hash") == file_hash:
            # Rehydrate cached chunks
            for chk in cached_entry["chunks"]:
                all_chunks.append(MinimalSource(**chk))
            continue

        # File changed or new: delete old chunks from ChromaDB (Bonus 3)
        if cached_entry:
            try:
                collection.delete(where={"file_path": str_path})
            except Exception:
                pass

        try:
            file_chunks = _chunk_file(file_path, max_chunk_size)
            all_chunks.extend(file_chunks)
            new_chunks.extend(file_chunks)

            manifest[str_path] = {
                "mtime": mtime,
                "hash": file_hash,
                "chunks": [c.model_dump(mode="json") for c in file_chunks],
            }
            manifest_updated = True
        except Exception as e:
            print(f"Skipping {file_path}: {e}", file=sys.stderr)

    # Save manifest ONCE at the end
    if manifest_updated or not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Step 3: Build & save BM25 lexical index
    lexical_index = build_lexical_index(all_chunks)
    save_lexical_index(lexical_index)

    # Step 4: Add new chunk embeddings to Chroma collection
    if new_chunks:
        vectors = build_semantic_index(new_chunks)
        save_semantic_index(vectors, new_chunks, collection)

    print(f"Ingestion complete! Indexed {len(all_chunks)} chunks under {PROCESSED_DIR}/")
```

