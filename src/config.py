from pathlib import Path

RAW_DIR: Path = Path("data/raw/vllm-0.10.1")
PROCESSED_DIR: Path = Path("data/processed")
INDEX_PATH: Path = PROCESSED_DIR / "lexical_index.pkl"


INDEX_DIR: str = "data/processed/"
RAW_DATA_DIR: str = "data/raw"


TOP_K: int = 10
MAX_CHUNK_SIZE: int = 2000

DATASET_DIR: str = "data/datasets/public/"


SEARCH_DATASET_PATH: str = (
    DATASET_DIR + "UnansweredQuestions/dataset_docs_public.json"
)
SEARCH_SAVE_DIRECTORY: str = "data/output/search_results/"


ANSWERED_DATASET_PATH: str = (
    DATASET_DIR + "AnsweredQuestions/dataset_docs_public.json"
)

ANSWER_SAVE_DIRECTORY: str = "data/output/search_results_and_answer/"

STUDENT_SEARCH_RESULTS_PATH: str = (
    SEARCH_SAVE_DIRECTORY + SEARCH_DATASET_PATH.split("/")[-1]
)


CACHE_PATH: str = "data/cache/query_cache.json"
FILE_METADATA = PROCESSED_DIR / "manifest.json"

QWEN_MODEL_ID: str = "Qwen/Qwen3-0.6B"
QWEN_MAX_NEW_TOKENS: int = 300
QWEN_TEMPERATURE: float = 0.0  # deterministic
