INDEX_DIR = "data/processed/"
RAW_DATA_DIR = "data/raw"


TOP_K = 10
MAX_CHUNK_SIZE = 2000
MIN_CHUNK_SIZE = 200

DATASET_DIR = "data/datasets/public/"


SEARCH_DATASET_PATH = (
    DATASET_DIR + "UnansweredQuestions/dataset_docs_public.json"
)


ANSWERED_DATASET_PATH = (
    DATASET_DIR + "AnsweredQuestions/dataset_docs_public.json"
)
SEARCH_SAVE_DIRECTORY = "data/output/search_results/"
ANSWER_SAVE_DIRECTORY = "data/output/search_results_and_answer/"

STUDENT_SEARCH_RESULTS_PATH = (
    SEARCH_SAVE_DIRECTORY + SEARCH_DATASET_PATH.split("/")[-1]
)
