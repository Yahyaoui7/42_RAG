from __future__ import annotations

from pathlib import Path
from typing import Any
import bm25s
from tqdm import tqdm

from src.config import INDEX_PATH, PROCESSED_DIR
from src.indexer import Index
from src.models import (
    MinimalSource,
    MinimalSearchResults,
    StudentSearchResults,
    UnansweredQuestion,
)


class Searcher:
    """Hybrid lexical and semantic search engine."""

    def __init__(self, index_path: Path = INDEX_PATH, k: int = 10) -> None:
        """Initialize indices and embedding models."""
        self.k = k
        self.index = Index()
        self.MiniLM_model = self.index.model
        self.index_lexical_data = self.index.load_index(index_path)

    def lexical_searcha(self, query: str) -> list[dict[str, Any]]:
        """Retrieve candidates using BM25 lexical search."""
        if self.k <= 0 or not query.strip():
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
                "id": str(self.index_lexical_data.chunks[i].id),
                "score": float(score),
                "chunk": self.index_lexical_data.chunks[i],
            }
            for i, score in zip(ranked_indices, score_indices)
        ]

    def semantic_search(
        self, query: str, path: Path = PROCESSED_DIR
    ) -> list[dict[str, Any]]:
        """Retrieve candidates using dense vector similarity in ChromaDB."""
        if self.k <= 0 or not query.strip():
            return []

        query_vector = self.MiniLM_model.encode(
            query, normalize_embeddings=True
        )

        results = self.index.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=self.k,
            include=["documents", "metadatas", "distances"],
        )

        ids_list = results.get("ids")
        docs_list = results.get("documents")
        metas_list = results.get("metadatas")
        dists_list = results.get("distances")

        if (
            not ids_list
            or not docs_list
            or not metas_list
            or not dists_list
            or not ids_list[0]
        ):
            return []

        ids = ids_list[0]
        docs = docs_list[0]
        metas = metas_list[0]
        dists = dists_list[0]

        final_return: list[dict[str, Any]] = []
        for i in range(len(ids)):
            metadata: Any = metas[i]
            chunk_metadata = MinimalSource(
                id=str(ids[i]),
                text=str(docs[i]),
                file_path=str(metadata["file_path"]),
                first_character_index=int(metadata["first_character_index"]),
                last_character_index=int(metadata["last_character_index"]),
            )

            chunk: dict[str, Any] = {
                "id": str(ids[i]),
                "score": float(dists[i]),
                "chunk": chunk_metadata,
            }
            final_return.append(chunk)

        return final_return

    def format_as_student_results(
        self, questions: list[UnansweredQuestion]
    ) -> StudentSearchResults:
        """Run batch search across question list and format response."""
        results: list[MinimalSearchResults] = []

        for q in tqdm(questions, desc="Searching questions"):
            final_result = self.get_resources(q.question)
            results.append(
                MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=final_result,
                )
            )
        return StudentSearchResults(search_results=results, k=self.k)

    def get_resources(self, query: str) -> list[MinimalSource]:
        """Perform hybrid retrieval combining BM25 and vector search."""
        lexical_result = self.lexical_searcha(query)
        semantic_result = self.semantic_search(query)
        return self.wsf_hybrid_reriever(lexical_result, semantic_result)

    def rrf_hybrid_retrieval(
        self,
        lexical_result: list[dict[str, Any]],
        semantic_result: list[dict[str, Any]],
    ) -> list[MinimalSource]:
        """Fuse rankings using Reciprocal Rank Fusion."""
        rrf_hybrid: dict[str, float] = {}
        chunks_by_id: dict[str, MinimalSource] = {}
        constant = 60

        for results in (lexical_result, semantic_result):
            for rank, chunk in enumerate(results, start=1):
                chunk_id = str(chunk["id"])
                chunks_by_id[chunk_id] = chunk["chunk"]

                score = 1.0 / (constant + rank)
                rrf_hybrid[chunk_id] = rrf_hybrid.get(chunk_id, 0.0) + score

        id_sorted = sorted(
            rrf_hybrid.items(), key=lambda item: item[1], reverse=True
        )

        return [chunks_by_id[chunk_id] for chunk_id, _ in id_sorted[: self.k]]

    def wsf_hybrid_reriever(
        self,
        lexical_result: list[dict[str, Any]],
        semantic_result: list[dict[str, Any]],
    ) -> list[MinimalSource]:
        """Fuse candidates using Weighted Score Fusion."""
        wsf_hybrid: dict[str, float] = {}
        chunks_by_id: dict[str, MinimalSource] = {}

        self.wsf_lexical(lexical_result)
        self.wsf_semantic(semantic_result)

        for result in (lexical_result, semantic_result):
            for chunk in result:
                chunk_id = str(chunk["id"])
                chunks_by_id[chunk_id] = chunk["chunk"]

                score = float(chunk["score"])
                wsf_hybrid[chunk_id] = wsf_hybrid.get(chunk_id, 0.0) + score

        id_sorted = sorted(
            wsf_hybrid.items(), key=lambda item: item[1], reverse=True
        )

        return [chunks_by_id[chunk_id] for chunk_id, _ in id_sorted[: self.k]]

    def wsf_lexical(
        self, lexical_result: list[dict[str, Any]], weight: float = 0.8
    ) -> None:
        """Min-max normalize lexical scores and apply weight."""
        if not lexical_result:
            return
        lex_min = min(float(chunk["score"]) for chunk in lexical_result)
        lex_max = max(float(chunk["score"]) for chunk in lexical_result)

        for chunk in lexical_result:
            if lex_max == lex_min:
                chunk["score"] = 0.0
            else:
                raw_score = float(chunk["score"])
                chunk["score"] = (raw_score - lex_min) / (lex_max - lex_min)

            chunk["score"] *= weight

    def wsf_semantic(
        self, semantic_result: list[dict[str, Any]], weight: float = 0.2
    ) -> None:
        """Convert distance to similarity and apply weight."""
        if not semantic_result:
            return
        for chunk in semantic_result:
            similarity = 1.0 - float(chunk["score"])
            chunk["score"] = similarity * weight
