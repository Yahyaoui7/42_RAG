# Detailed Issue Analysis: `src/caching.py` (Bonus 4)

**File:** [`src/caching.py`](file:///home/nyahyaou/Desktop/RAG/src/caching.py)  
**Severity:** 🔵 **BONUS / MEDIUM** (Incomplete implementation of Bonus 4, crash risk on missing directory)

---

## 1. Overview of Problems

| Bug # | Type | Location | Impact |
|---|---|---|---|
| **6.1** | Incomplete Bonus 4 Scope | Lines 18–36 | Only caches generated LLM answers; **does NOT cache search query results**, violating Bonus 4 requirement. |
| **6.2** | Crash on Missing Cache Directory | Line 35 | `open(self.cache_path, "w")` crashes with `FileNotFoundError` if `data/cache/` does not already exist. |
| **6.3** | Typo in Cache Filename | `src/config.py` line 35 | `query_cach.json` (missing 'e' in cache). |
| **6.4** | Missing Type Annotations & Docstrings | Lines 8, 18, 27, 30 | Violates flake8 and mypy strict guidelines. |

---

## 2. In-Depth Root Cause Analysis

### Issue 6.1: Incomplete Bonus 4 Specification
Bonus 4 in Chapter IX (page 23) explicitly states:
> *"4. Caching: cache the index and query results to speed up cold start and repeated queries."*

Currently:
* You cache the index to disk (via pickle for BM25 and ChromaDB for vectors).
* You cache the **answers** in `Caching.save_query_and_answer()`.
* **BUT you never cache search results!** If a user runs `search "how to load lora adapter"` twice, the entire hybrid search runs again.

To fully validate Bonus 4, the caching mechanism should support:
1. `get_search_cache(query, k)` / `save_search_cache(query, k, results)`.
2. `get_answer_cache(query, k)` / `save_answer_cache(query, k, answer)`.

---

### Issue 6.2: Missing Directory Creation
In `save_query_and_answer`:
```python
with open(self.cache_path, "w") as file:
    json.dump(self.cache, file, indent=4)
```
If `data/cache` does not exist on a fresh clone or evaluator machine, this line immediately crashes:
`FileNotFoundError: [Errno 2] No such file or directory: 'data/cache/query_cach.json'`.

#### Fix:
Ensure parent directory exists before writing:
```python
path = Path(self.cache_path)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(self.cache, indent=2), encoding="utf-8")
```

---

## 3. Corrected Implementation for `src/caching.py`

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.config import CACHE_PATH
from src.models import MinimalSource


class Caching:
    """Manages persistent disk caching for search results and generated answers (Bonus 4)."""

    def __init__(self, top_k: int = 10, cache_path: str = CACHE_PATH) -> None:
        """Initialize cache store with support for search and answer results."""
        self.top_k = int(top_k)
        self.cache_path = Path(cache_path)
        self.cache: Dict[str, Any] = {"answers": {}, "search": {}}

        if self.cache_path.is_file():
            try:
                content = self.cache_path.read_text(encoding="utf-8")
                loaded = json.loads(content or "{}")
                if "answers" in loaded and "search" in loaded:
                    self.cache = loaded
                else:
                    # Backward compatibility with flat answer dictionary
                    self.cache["answers"] = loaded
            except Exception:
                pass

    def _normalize(self, query: str) -> str:
        """Normalize query by stripping whitespace and lowering case."""
        return " ".join(query.lower().split())

    def _make_key(self, query: str) -> str:
        """Build cache key incorporating query and k value."""
        return f"{self._normalize(query)} [top_k={self.top_k}]"

    def get_answer_from_cache(self, query: str) -> Optional[str]:
        """Retrieve cached LLM answer if available."""
        key = self._make_key(query)
        return self.cache.get("answers", {}).get(key)

    def save_query_and_answer(self, query: str, answer: str) -> None:
        """Save generated answer to disk cache."""
        key = self._make_key(query)
        self.cache.setdefault("answers", {})[key] = answer
        self._flush()

    def get_search_from_cache(self, query: str) -> Optional[List[MinimalSource]]:
        """Retrieve cached search source locations if available."""
        key = self._make_key(query)
        cached_items = self.cache.get("search", {}).get(key)
        if cached_items is None:
            return None
        return [MinimalSource.model_validate(item) for item in cached_items]

    def save_search_results(self, query: str, results: List[MinimalSource]) -> None:
        """Save retrieved search sources to disk cache."""
        key = self._make_key(query)
        self.cache.setdefault("search", {})[key] = [
            item.model_dump(mode="json") for item in results
        ]
        self._flush()

    def _flush(self) -> None:
        """Flush in-memory cache to disk safely."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, indent=2), encoding="utf-8")
        except Exception:
            pass
```

