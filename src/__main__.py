from __future__ import annotations
import fire, sys
from src.indexer import Index
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
from pathlib import Path

from src.prompts import Prompt
from src.generate import Generate
from src.caching import Caching


class CLI:
    """Command-line interface for the RAG system built with Python Fire."""

    def index(self, max_chunk_size: int = MAX_CHUNK_SIZE) -> None:
        """Ingest data/raw/ and build indices under data/processed/.

        Args:
            max_chunk_size: Maximum character length per chunk (<= 2000).
        """
        try:
            index = Index()
            if max_chunk_size <= 0:
                print(
                    "Error: max_chunk_size must be positive.", file=sys.stderr
                )
                return
            index.build_and_save_from_raw(max_chunk_size)
        except Exception as e:
            print(f"Error indexing corpus: {e}", file=sys.stderr)

    def search(self, query: str = "", k: int = TOP_K) -> None:
        """Return top-k sources for a single query.

        Args:
            query: Question or keywords to search.
            k: Number of source locations to retrieve.
        """
        try:
            if not query or not query.strip() or k <= 0:
                return

            searcher = Searcher(k=k)
            results = searcher.get_resources(query)

            for chunk in results:
                # Exact format required: [start:end] without space
                print(
                    f"{chunk.file_path} [{chunk.first_character_index}:{chunk.last_character_index}]"
                )
        except Exception as e:
            print(f"Error during search: {e}", file=sys.stderr)

    def search_dataset(
        self,
        dataset_path: str = SEARCH_DATASET_PATH,
        save_directory: str = SEARCH_SAVE_DIRECTORY,
        k: int = TOP_K,
    ) -> None:
        """Run search across an entire question dataset and output JSON results.

        Args:
            dataset_path: Path to dataset JSON file.
            save_directory: Directory where student search results will be saved.
            k: Number of results to retrieve per question.
        """
        try:
            path = Path(dataset_path)
            if not path.is_file():
                print(
                    f"Error: Dataset file not found at {dataset_path}",
                    file=sys.stderr,
                )
                return

            raw_content = path.read_text(encoding="utf-8")
            questions = RagDataset.model_validate_json(
                raw_content
            ).rag_questions

            searcher = Searcher(k=k)
            result = searcher.format_as_student_results(questions)

            out_dir = Path(save_directory)
            out_dir.mkdir(parents=True, exist_ok=True)

            output_file = out_dir / path.name
            output_file.write_text(
                result.model_dump_json(indent=2), encoding="utf-8"
            )
            print(f"Saved student_search_results to {output_file}")
        except Exception as e:
            print(f"Error processing search_dataset: {e}", file=sys.stderr)

    def answer(self, query: str = "", k: int = 10) -> None:
        """Answer a single query using retrieved context.

        Args:
            query: Question text.
            k: Number of sources to retrieve for augmentation.
        """
        try:

            if not query or not query.strip() or k <= 0:
                print("answer: I don't know.")
                return

            cache = Caching(top_k=k)
            cached = cache.get_answer_from_cache(query)
            if cached:
                print("answer:", cached)
                return

            searcher = Searcher(k=k)
            sources = searcher.get_resources(query)
            prompt = Prompt().build_prompt(query, sources)

            generator = Generate()
            answer_text = generator.generate_answer(prompt)

            cache.save_query_and_answer(query, answer_text)
            print("answer:", answer_text)
        except Exception as e:
            print(f"Error generating answer: {e}", file=sys.stderr)

    def answer_dataset(
        self,
        student_search_results_path: str = STUDENT_SEARCH_RESULTS_PATH,
        save_directory: str = ANSWER_SAVE_DIRECTORY,
    ) -> None:
        """Generate answers for all questions in a student search results file.

        Args:
            student_search_results_path: Path to the search results JSON.
            save_directory: Output directory for the answered JSON file.
        """
        try:
            input_path = Path(student_search_results_path)
            if not input_path.is_file():
                print(
                    f"Error: Search results file not found at {input_path}",
                    file=sys.stderr,
                )
                return

            content = input_path.read_text(encoding="utf-8")
            student_result = StudentSearchResults.model_validate_json(content)

            out_dir = Path(save_directory)
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / input_path.name

            generator = Generate()
            generator.answer_dataset(student_result, output_path)
            print(f"Saved student_search_results_and_answer to {output_path}")
        except Exception as e:
            print(f"Error running answer_dataset: {e}", file=sys.stderr)

    def evaluate(
        self,
        student_search_results_path: str = STUDENT_SEARCH_RESULTS_PATH,
        dataset_path: str = ANSWERED_DATASET_PATH,
    ) -> None:
        """Evaluate Recall@k scores against ground truth.

        Args:
            student_search_results_path: Path to student search results JSON.
            dataset_path: Path to ground-truth AnsweredQuestions JSON.
        """
        try:
            if not Path(student_search_results_path).is_file():
                print(
                    f"Error: Missing search results at {student_search_results_path}",
                    file=sys.stderr,
                )
                return
            if not Path(dataset_path).is_file():
                print(
                    f"Error: Missing ground truth dataset at {dataset_path}",
                    file=sys.stderr,
                )
                return

            evaluator = Evaluator(student_search_results_path, dataset_path)
            scores = evaluator.get_score()

            print("\nEvaluation Results")
            print("========================================")
            for k_val, val in scores.items():
                print(f"Recall@{k_val}: {val:.3f} ({val * 100:.1f}%)")
        except Exception as e:
            print(f"Error during evaluation: {e}", file=sys.stderr)

    def serve(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Run the local HTTP API server (Bonus 5).

        Args:
            host: IP address or host to bind.
            port: Port number to listen on.
        """
        try:
            import uvicorn
            from src.server import app

            print(f"Starting Local HTTP API on http://{host}:{port} ...")
            uvicorn.run(app, host=host, port=int(port))
        except Exception as e:
            print(f"Error starting server: {e}", file=sys.stderr)


def main() -> None:
    """CLI entry point using Python Fire."""
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
