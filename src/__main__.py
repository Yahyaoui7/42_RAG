from __future__ import annotations
import fire, sys
from src.indexer import build_and_save_from_raw
from src.config import (
    MAX_CHUNK_SIZE,
    SEARCH_DATASET_PATH,
    SEARCH_SAVE_DIRECTORY,
    TOP_K,
    STUDENT_SEARCH_RESULTS_PATH,
    ANSWER_SAVE_DIRECTORY,
    ANSWERED_DATASET_PATH,
)
from src.retriever import Searcher
from src.models import MinimalSearchResults, StudentSearchResults, RagDataset
from src.evaluator import Evaluator
import uuid
from pathlib import Path
import json
from src.prompts import Prompt
from src.generate import Generate
from src.caching import Caching


class CLI:

    def index(self, max_chunk_size: int = MAX_CHUNK_SIZE) -> None:
        try:
            build_and_save_from_raw(max_chunk_size)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    def search(self, query: str, k: int = 10) -> None:
        searcher = Searcher(k=k)  # loads data/processed/index.pkl once
        result = searcher.get_resources(query)

        for chunk in result:
            print(
                f"{chunk.file_path} [{chunk.first_character_index}: {chunk.last_character_index}]"
            )

    def search_dataset(
        self,
        dataset_path=SEARCH_DATASET_PATH,
        save_directory=SEARCH_SAVE_DIRECTORY,
        k: int = TOP_K,
    ):
        questions = RagDataset.model_validate_json(
            Path(dataset_path).read_text()
        ).rag_questions

        searcher = Searcher(k=k)  # load index.pkl ONCE, not 100x

        result = searcher.format_as_student_results(
            questions
        )  # inside: for q in tqdm(questions): searcher.search(q.question, k) -> uses your cache tuple(tokens) -> ranked[:k]
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        (Path(save_directory) / Path(dataset_path).name).write_text(
            result.model_dump_json(indent=2)
        )

    def answer(self, query, k):
        cache = Caching(k)
        cache_answer = cache.get_answer_from_cache(query)
        if cache_answer:
            print("answer: ", cache_answer)
        else:

            searcher = Searcher()
            model = Generate()

            sources = searcher.get_resources(query)
            prompt = Prompt().build_prompt(query, sources)
            answer = model.generate_answer(prompt)

            cache.save_query_and_answer(query, answer)
            print("answer: ", answer)

    def answer_dataset(
        self,
        student_search_results_path=STUDENT_SEARCH_RESULTS_PATH,
        save_directory=ANSWER_SAVE_DIRECTORY,
    ):

        student_result = StudentSearchResults.model_validate_json(
            Path(student_search_results_path).read_text()
        )
        genreter = Generate()
        genreter.answer_dataset(
            student_result,
            (Path(save_directory) / Path(student_search_results_path).name),
        )

    def evaluate(
        self,
        student_search_results_path=STUDENT_SEARCH_RESULTS_PATH,
        dataset_path=ANSWERED_DATASET_PATH,
    ):
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


# import hashlib

# def get_file_hash(file_path):
#     sha256 = hashlib.sha256()

#     with open(file_path, "rb") as file:
#         while chunk := file.read(4096):
#             sha256.update(chunk)

#     return sha256.hexdigest()
