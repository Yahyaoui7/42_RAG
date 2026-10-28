from pathlib import Path
import json
import numpy as np
from src.chunking.base import chunk_to_minimal_source, Chunk
from src.config import INDEX_PATH, SAVE_CHUNKS_PATH
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
    def __init__(self, index_path: Path = INDEX_PATH, k: int = 10):
        self.MiniLM_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.k = k
        self.index_lexical_data = load_index(index_path)
        self.semantic_index = load_index(
            Path(index_path).with_name("semantic_index.npy")
        )
        self.chunks_save = [
            MinimalSource.model_validate(chunk)
            for chunk in json.loads(Path(SAVE_CHUNKS_PATH).read_text())
        ]

    def lexical_searcha(self, query: str) -> list[MinimalSource]:

        if self.k <= 0:
            return []
        tokens = bm25s.tokenize([query])

        if not tokens:
            return []
        if (
            self.index_lexical_data.bm25 is None
            or not self.index_lexical_data.chunks
        ):
            return []

        result, scores = self.index_lexical_data.bm25.retrieve(
            tokens, k=self.k
        )
        ranked_indices = result[0].tolist()
        score_indices = scores[0].tolist()

        return [
            {
                "id": self.index_lexical_data.chunks[i].id,
                "score": float(score),
                "chunk": self.index_lexical_data.chunks[i],
            }
            for i, score in zip(ranked_indices, score_indices)
        ]

    def semantic_search(self, query: str):

        embading_query = self.MiniLM_model.encode(
            query, normalize_embeddings=True
        )
        scores = self.semantic_index @ embading_query
        top_indices = np.argsort(scores)[::-1][: self.k]
        return [
            {
                "id": self.index_lexical_data.chunks[i].id,
                "score": float(scores[i]),
                "chunk": self.chunks_save[i],
            }
            for i in top_indices
        ]

    def format_as_student_results(
        self, questions: list[UnansweredQuestion]
    ) -> StudentSearchResults:
        # reuse self.index_lexical_data + self.search -> loads ONCE, 100x faster
        results = []

        for q in tqdm(questions):  # add tqdm here later
            lexical_result = self.lexical_searcha(q.question)
            semantic_result = self.semantic_search(q.question)
            final_result = self.wsf_hybrid_reriever(
                lexical_result, semantic_result
            )
            results.append(
                MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=final_result,
                )
            )
        return StudentSearchResults(search_results=results, k=self.k)

    def rrf_hybrid_retrieval(self, lexical_result, semantic_result):
        rrf_hybrid = {}
        chunks_by_id = {}
        constant = 60

        for results in (lexical_result, semantic_result):
            for rank, chunk in enumerate(results, start=1):
                chunk_id = chunk["id"]
                chunks_by_id[chunk_id] = chunk["chunk"]

                score = 1 / (constant + rank)
                rrf_hybrid[chunk_id] = rrf_hybrid.get(chunk_id, 0) + score

        id_sorted = sorted(
            rrf_hybrid.items(), key=lambda item: item[1], reverse=True
        )

        final_rank = [
            chunks_by_id[chunk_id] for chunk_id, score in id_sorted[: self.k]
        ]

        return final_rank

    def wsf_hybrid_reriever(self, lexical_result, semantic_result):
        wsf_hybrid = {}
        chunks_by_id = {}

        self.wsf_lexical(lexical_result)
        self.wsf_semantic(semantic_result)

        for result in (lexical_result, semantic_result):
            for chunk in result:
                chunk_id = chunk["id"]
                chunks_by_id[chunk_id] = chunk["chunk"]

                score = chunk["score"]
                wsf_hybrid[chunk_id] = wsf_hybrid.get(chunk_id, 0) + score
        id_sorted = sorted(
            wsf_hybrid.items(), key=lambda item: item[1], reverse=True
        )

        final_rank = [
            chunks_by_id[chunk_id] for chunk_id, score in id_sorted[: self.k]
        ]
        return final_rank

    def wsf_lexical(self, lexical_result):
        lex_min = min([chunk["score"] for chunk in lexical_result])
        lex_max = max([chunk["score"] for chunk in lexical_result])

        if lex_max == lex_min:
            for chunk in lexical_result:
                chunk["score"] = 0.0
        else:
            for chunk in lexical_result:
                chunk["score"] = (
                    (chunk["score"] - lex_min) / (lex_max - lex_min)
                ) * 0.5

    def wsf_semantic(self, semantic_result):
        for chunk in semantic_result:
            chunk["score"] = chunk["score"] * 0.5
