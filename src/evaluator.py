import json
from pathlib import Path
from src.models import MinimalSource


class Evaluator:
    def __init__(self, student_search_results_path, dataset_path): #check if teh load became field
        self.student_result = json.loads(
            Path(student_search_results_path).read_text()
        )
        self.dataset_result = json.loads(Path(dataset_path).read_text())
        self.result_dict = {
            question["question_id"]: question["retrieved_sources"]
            for question in self.student_result["search_results"]
        }

    def check_is_valid(self, raw_source, student_source):
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

        first_nestad = max(
            raw_source.first_character_index,
            student_source.first_character_index,
        )
        last_nestad = min(
            raw_source.last_character_index,
            student_source.last_character_index,
        )
        intersection = max(0, last_nestad - first_nestad)
        union = all_range
        # print(f"{(intersection / union):2f}")
        return (intersection / union) >= 0.05

    def check_top_k(self, raw_source, student_sources, k):
        for source in student_sources[:k]:
            if self.check_is_valid(raw_source, source):
                return True

        return False

    def get_score(self):

        questions = self.dataset_result["rag_questions"]
        checker = [1, 3, 5, 10]
        score_dict = {}
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
            score = result / len(questions) if questions else 0
            score_dict[k] = float(score)
        return score_dict

        # return score
