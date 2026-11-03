from __future__ import annotations

import json
from pathlib import Path
from src.models import MinimalSource


class Evaluator:
    """Evaluates student retrieval Recall@k against reference dataset."""

    def __init__(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        """Load search results and ground truth dataset."""
        self.student_result = json.loads(
            Path(student_search_results_path).read_text(encoding="utf-8")
        )
        self.dataset_result = json.loads(
            Path(dataset_path).read_text(encoding="utf-8")
        )
        self.result_dict = {
            question["question_id"]: question["retrieved_sources"]
            for question in self.student_result["search_results"]
        }

    def check_is_valid(
        self, raw_source: MinimalSource, student_source: MinimalSource
    ) -> bool:
        """Check if student source overlaps reference span with IoU >= 0.05."""
        if raw_source.file_path != student_source.file_path:
            return False
        first = min(
            raw_source.first_character_index,
            student_source.first_character_index,
        )
        last = max(
            raw_source.last_character_index,
            student_source.last_character_index,
        )
        all_range = last - first
        if all_range <= 0:
            return False

        first_nested = max(
            raw_source.first_character_index,
            student_source.first_character_index,
        )
        last_nested = min(
            raw_source.last_character_index,
            student_source.last_character_index,
        )
        intersection = max(0, last_nested - first_nested)
        union = all_range
        return (intersection / union) >= 0.05

    def check_top_k(
        self,
        raw_source: MinimalSource,
        student_sources: list[MinimalSource],
        k: int,
    ) -> bool:
        """Check if any of top-k student sources matches the ground truth."""
        for source in student_sources[:k]:
            if self.check_is_valid(raw_source, source):
                return True

        return False

    def get_score(self) -> dict[int, float]:
        """Compute Recall@k for k in [1, 3, 5, 10]."""
        questions = self.dataset_result["rag_questions"]
        checker = [1, 3, 5, 10]
        score_dict: dict[int, float] = {}
        for k in checker:
            result = 0
            for question in questions:
                raw_source = MinimalSource.model_validate(
                    question["sources"][0]
                )
                student_source = [
                    MinimalSource.model_validate(source)
                    for source in self.result_dict[question["question_id"]]
                ]
                if self.check_top_k(raw_source, student_source, k):
                    result += 1
            score = result / len(questions) if questions else 0.0
            score_dict[k] = float(score)
        return score_dict
