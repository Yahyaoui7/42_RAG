from __future__ import annotations
import fire, sys
from src.indexer import build_and_save_from_raw
from src.config import DEFAULT_MAX_CHUNK_SIZE
from src.retriever import Searcher
from src.models import MinimalSearchResults, StudentSearchResults, RagDataset
from src.evaluator import Evaluator
import uuid
from pathlib import Path
import json
from src.prompts import Prompt
from src.generate import Generate


class CLI:

    def index(self, max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE) -> None:
        try:
            build_and_save_from_raw(max_chunk_size)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    def search(self, query: str, k: int = 10) -> None:
        searcher = Searcher()  # loads data/processed/index.pkl once
        sources = searcher.search(query, k)
        for chunk in sources:
            print(
                f"{chunk.file_path} [{chunk.first_character_index}: {chunk.last_character_index}]"
            )

    def search_dataset(
        self, dataset_path: str, save_directory: str, k: int = 10
    ):
        questions = RagDataset.model_validate_json(
            Path(dataset_path).read_text()
        ).rag_questions
        searcher = Searcher()  # load index.pkl ONCE, not 100x

        result = searcher.format_as_student_results(
            questions, k
        )  # inside: for q in tqdm(questions): searcher.search(q.question, k) -> uses your cache tuple(tokens) -> ranked[:k]
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        (Path(save_directory) / Path(dataset_path).name).write_text(
            result.model_dump_json(indent=2)
        )

    def answer(self, query, k):
        searcher = Searcher()
        model = Generate()

        sources = searcher.search(query, k)
        prompt = Prompt().build_prompt(query, sources)
        answer = model.generate_answer(prompt)
        print("answer: ", answer)

    def answer_dataset(self, student_search_results_path, save_directory):

        student_result = StudentSearchResults.model_validate_json(
            Path(student_search_results_path).read_text()
        )
        genreter = Generate()
        genreter.answer_dataset(student_result, save_directory)

    def evaluate(self, student_search_results_path: Path, dataset_path: Path):
        print("\nEvaluation Results")
        print("==" * 10)
        evaluater = Evaluator(student_search_results_path, dataset_path)

        score = evaluater.get_score()
        print("Questions evaluated:")
        for k, v in score.items():
            print(f"Recall@{k}: {v:.3f} ({float(v * 100):.1f}%)")


def main() -> None:
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
