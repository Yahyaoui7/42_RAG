from __future__ import annotations

import json
from pathlib import Path
from src.config import CACHE_PATH


class Caching:
    """Cache query answers to accelerate warm inference."""

    def __init__(self, top_k: int, cache_path: str = CACHE_PATH) -> None:
        """Initialize cache store."""
        self.cache_path = Path(cache_path)
        self.top_k = top_k
        self.cache: dict[str, str] = {}

        if Path(cache_path).exists():
            content = Path(cache_path).read_text(encoding="utf-8")
            data = json.loads(content or "{}")
            if isinstance(data, dict):
                self.cache = {str(k): str(v) for k, v in data.items()}

    def get_answer_from_cache(self, query: str) -> str | None:
        """Retrieve answer from cache if present."""
        query_norm = self.normalize(query)
        query_norm += f" [top_k={self.top_k}]"

        if query_norm in self.cache:
            return self.cache[query_norm]
        return None

    def normalize(self, query: str) -> str:
        """Normalize query whitespace and case."""
        return " ".join(query.lower().split())

    def save_query_and_answer(self, query: str, answer: str) -> None:
        """Persist query and answer to cache file."""
        norm_key = self.normalize(query) + f" [top_k={self.top_k}]"
        self.cache[norm_key] = answer

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as file:
            json.dump(self.cache, file, indent=4)
