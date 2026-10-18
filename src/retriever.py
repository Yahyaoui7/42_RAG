from pathlib import Path
import numpy as np
from src.chunking.base import chunk_to_minimal_source
from src.config import INDEX_PATH
from src.indexer import IndexData, load_index
from src.models import (
    MinimalSource,
    MinimalSearchResults,
    StudentSearchResults,
    UnansweredQuestion,
)
from tqdm import tqdm
import bm25s


class Searcher:
    def __init__(self, index_path: Path = INDEX_PATH):
        self.index_data = load_index(index_path)

    def search(self, query: str, k: int = 10) -> list[MinimalSource]:

        if k <= 0:
            return []
        tokens = bm25s.tokenize([query])

        if not tokens:
            return []
        if self.index_data.bm25 is None or not self.index_data.chunks:
            return []

        result, scores = self.index_data.bm25.retrieve(tokens, k=k)

        ranked_indices = result[0].tolist()

        return [
            chunk_to_minimal_source(self.index_data.chunks[i])
            for i in ranked_indices
        ]

    def search_one(
        self, query: str, index_path: Path = INDEX_PATH, k: int = 5
    ) -> list[MinimalSource]:
        # for CLI single query, reload if different index_path
        if index_path != INDEX_PATH:
            self.index_data = load_index(index_path)
        return self.search(query, k)

    def format_as_student_results(
        self, questions: list[UnansweredQuestion], k: int
    ) -> StudentSearchResults:
        # reuse self.index_data + self.search -> loads ONCE, 100x faster
        results = []

        for q in tqdm(questions):  # add tqdm here later
            sources = self.search(q.question, k)
            results.append(
                MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=sources,
                )
            )
        return StudentSearchResults(search_results=results, k=k)
