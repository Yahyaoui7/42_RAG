import json
from pathlib import Path
from src.config import CACHE_PATH


class Caching:

    def __init__(self, top_k, cache_path=CACHE_PATH):
        self.cache_path = cache_path
        self.top_k = top_k

        if Path(cache_path).exists():
            content = Path(cache_path).read_text()
            self.cache = json.loads(content or "{}")
        else:
            self.cache = {}

    def get_answer_from_cache(self, query):
        query_norm = self.normalize(query)
        query_norm += f" [top_k={self.top_k}]"

        if query_norm in self.cache:
            return self.cache[query_norm]
        else:
            return None

    def normalize(self, query):
        return " ".join(query.lower().split())

    def save_query_and_answer(self, query, answer):
        query = self.normalize(query) + f" [top_k={self.top_k}]"

        self.cache[query] = answer

        with open(self.cache_path, "w") as file:
            json.dump(self.cache, file, indent=4)
