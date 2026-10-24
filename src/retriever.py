from pathlib import Path
import json
import numpy as np
from src.chunking.base import chunk_to_minimal_source, Chunk
from src.config import INDEX_PATH, CHUNKS_PATH
from src.indexer import IndexData, load_index
from src.models import (
    MinimalSource,
    MinimalSearchResults,
    StudentSearchResults,
    UnansweredQuestion,
)
from tqdm import tqdm
import bm25s
import numpy as np
from sentence_transformers import SentenceTransformer


class Searcher:
    def __init__(self, index_path: Path = INDEX_PATH):
        self.index_lexical_data = load_index(index_path)
        self.semantic_index = load_index(
            Path(index_path).with_name("semantic_index.npy")
        )
        self.chunks_save = [
            MinimalSource.model_validate(chunk)
            for chunk in json.loads(Path(CHUNKS_PATH).read_text())
        ]

    def lexical_searcha(self, query: str, k: int = 10) -> list[MinimalSource]:

        if k <= 0:
            return []
        tokens = bm25s.tokenize([query])

        if not tokens:
            return []
        if (
            self.index_lexical_data.bm25 is None
            or not self.index_lexical_data.chunks
        ):
            return []

        result, scores = self.index_lexical_data.bm25.retrieve(tokens, k=k)
        ranked_indices = result[0].tolist()

        self.score_bm25 = [
            (self.index_lexical_data.chunks[i].text, score)
            for i, score in zip(ranked_indices, scores)
        ]
        for d in self.score_bm25:
            print((d[0]))
            print("\n-------------\n")
        # data = [
        #     chunk_to_minimal_source(self.index_lexical_data.chunks[i])
        #     for i in ranked_indices
        # ]

    def semantic_search(self, query: str, k: int = 10):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embading_query = model.encode(query, normalize_embeddings=True)
        scores = self.semantic_index @ embading_query
        top_indices = np.argsort(scores)[::-1][:k]
        result = [(self.chunks_save[i].text, scores[i]) for i in top_indices]
        return result

    def format_as_student_results(
        self, questions: list[UnansweredQuestion], k: int
    ) -> StudentSearchResults:
        # reuse self.index_lexical_data + self.search -> loads ONCE, 100x faster
        results = []

        for q in tqdm(questions):  # add tqdm here later
            sources = self.lexical_searcha(q.question, k)

            results.append(
                MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=sources,
                )
            )
        return StudentSearchResults(search_results=results, k=k)

    def hybrid_retrieval(bm25_score: tuple, semantic_scor: tuple):
        pass
