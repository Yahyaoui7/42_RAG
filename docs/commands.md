# Commands, data layout and workflow

Everything needed to run the system, what it produces, and how the evaluation
is driven.

---

## 1. Install

```bash
uv sync          # installs the dependencies from uv.lock
```

Requirements: Python **3.10+**, `uv`. The generation step downloads
`Qwen/Qwen3-0.6B` from the Hugging Face hub on first use (~1.5 GB); the
indexing step downloads `all-MiniLM-L6-v2` (~90 MB).

---

## 2. The six commands

All commands are invoked as `uv run python -m src <command>`. Fire accepts both
the long form (`--dataset_path`) and the short alias (`-d`).

### `index` — build the index

```bash
uv run python -m src index --max_chunk_size 2000
```

Scans `data/raw/vllm-0.10.1`, chunks every `.py` / `.md` / `.txt` file, builds
the BM25 index and the semantic index, and writes:

```text
data/processed/lexical_index.pkl     # IndexData: bm25 + chunks + tokenized corpus
data/processed/semantic_index.npy    # normalized MiniLM embeddings
```

Measured on the development machine: **1952 files → 14 458 chunks**, largest
chunk exactly 2000 characters, **197 s** in total — about 4 s for the lexical
index and 193 s for the semantic encoding.

> The chunk size is a ceiling, not a target: smaller chunks are allowed and
> improve the precision of the retrieved ranges. The value must never exceed
> **2000**, because the grader rejects any source wider than its
> `--max_context_length`.

### `search` — one question

```bash
uv run python -m src search "how do I enable LoRA adapters" --k 3
```

```text
data/raw/vllm-0.10.1/docs/features/lora.md [4075: 4693]
data/raw/vllm-0.10.1/docs/features/lora.md [1879: 2070]
```

Each line is `file_path [first_character_index: last_character_index]` — the
exact source location of a retrieved chunk.

### `search_dataset` — a whole dataset

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/public/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results
```

Writes `data/output/search_results/dataset_docs_public.json` — a
`StudentSearchResults` file. The output file keeps the name of the input
dataset, so several datasets can be written into the same directory as long as
their names differ.

### `answer` — one question

```bash
uv run python -m src answer "What is vLLM?" --k 10
```

Prints `answer:  <the generated answer>`. The model is loaded once per
invocation (~10 s) and the answer is greedy, therefore reproducible.

### `answer_dataset` — a whole search-results file

```bash
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer
```

Writes a `StudentSearchResultsAndAnswer` file. A progress bar tracks the
batch.

> **Known bug:** the file is not written — `src/generate.py` builds the output
> path from the pydantic model instead of the input file name and raises
> `TypeError`. See limitation 1 in
> [`architecture.md`](architecture.md#11-known-limitations).

### `evaluate` — local recall@k

```bash
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/dataset_docs_public.json \
  --dataset_path data/datasets/public/AnsweredQuestions/dataset_docs_public.json
```

```text
Evaluation Results
====================
Questions evaluated:
Recall@1: 0.590 (59.0%)
Recall@3: 0.780 (78.0%)
Recall@5: 0.810 (81.0%)
Recall@10: 0.890 (89.0%)
```

This is a local approximation for iteration. The official numbers come from
the `moulinette`.

---

## 3. Command summary

| Command | Required | Optional | Reads | Writes |
|---|---|---|---|---|
| `index` | – | `--max_chunk_size` | `data/raw/vllm-0.10.1` | `data/processed/*.pkl`, `*.npy` |
| `search` | `query` | `--k` | the index | stdout |
| `search_dataset` | – | `--dataset_path`, `--save_directory`, `--k` | a dataset + the index | `StudentSearchResults` JSON |
| `answer` | `query` | `--k` | the index | stdout |
| `answer_dataset` | – | `--student_search_results_path`, `--save_directory` | a `StudentSearchResults` JSON | `StudentSearchResultsAndAnswer` JSON |
| `evaluate` | – | `--student_search_results_path`, `--dataset_path` | results + ground truth | stdout |

Every path is a CLI argument and has a default in `src/config.py`; nothing is
hard-coded inside the functions.

---

## 4. Data layout on disk

```text
data/
├── raw/
│   └── vllm-0.10.1/                 # the indexed corpus
│       ├── docs/                    # Markdown documentation
│       ├── vllm/                    # the Python package
│       └── ...
├── processed/
│   ├── lexical_index.pkl            # BM25 index + chunks
│   └── semantic_index.npy           # MiniLM embeddings
├── datasets/
│   ├── public/
│   │   ├── UnansweredQuestions/     # dataset_docs_public.json, dataset_code_public.json
│   │   └── AnsweredQuestions/       # same names, with the reference sources
│   └── private/
│       ├── UnansweredQuestions/
│       └── AnsweredQuestions/
└── output/
    ├── search_results/              # output of search_dataset
    └── search_results_and_answer/   # output of answer_dataset
```

The `UnansweredQuestions` and `AnsweredQuestions` datasets share their file
names on purpose: the first holds the questions, the second holds the same
questions with their reference sources. They are always kept in separate
directories.

---

## 5. Output format

`search_dataset` produces a `StudentSearchResults`:

```json
{
  "search_results": [
    {
      "question_id": "17526382-7764-4120-b5e8-2d3726b8a4da",
      "question": "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?",
      "retrieved_sources": [
        {
          "text": "### Using API Endpoints\n\nLoading a LoRA Adapter:\n...",
          "file_path": "data/raw/vllm-0.10.1/docs/features/lora.md",
          "first_character_index": 4695,
          "last_character_index": 6098
        }
      ]
    }
  ],
  "k": 10
}
```

`answer_dataset` produces the same structure with an `answer` field added to
each entry (`StudentSearchResultsAndAnswer`).

- `file_path` is **verbatim** the corpus path, relative to the repository
  root, with forward slashes. The grader compares it literally — a different
  prefix never matches.
- `first_character_index` / `last_character_index` are the character range of
  the chunk inside that file, and the chunk text is exactly
  `file_content[first:last]`.
- `text` is an optional extra field (default `""`) that carries the chunk
  content so the answer stage does not have to re-read the file. The grader
  accepts it.

---

## 6. Makefile targets

```bash
make install   # pip install uv && uv sync
make run       # index + search_dataset + evaluate on docs, then on code
make debug     # python -m pdb -m src
make clean     # pyclean + remove .mypy_cache
make lint      # flake8 . && mypy . (with the mandatory flags)
```

`make run` is the fastest way to see the whole pipeline working locally; it
uses the public datasets and prints both recall reports.

---

## 7. Evaluation workflow

The grader drives the project with three scripts in `exams/scripts/`.

### Retrieval

```bash
./exams/scripts/exam_retrieval.sh --student-path . --moulinette-path ./moulinette-ubuntu
```

| Test | Budget | Measured |
|---|---|---|
| Indexing | ≤ 300 s | 197 s (4 s lexical + 193 s semantic encoding) |
| Retrieval of 200 questions | ≤ 90 s | ~5 s |
| Docs recall@5 | ≥ 80 % | 81 % (public) / 85 % (private) |
| Code recall@5 | ≥ 50 % | 55.6 % (public) / 50 % (private) |

Under the hood it runs:

```bash
uv run python -m src index --max_chunk_size 2000
uv run python -m src search_dataset --dataset_path <private/unanswered> --k 10 --save_directory <dir>
./moulinette-ubuntu evaluate_student_search_results <results.json> <truth.json> \
    --k 10 --max_context_length 2000 --threshold 0.8
```

### Answers

```bash
./exams/scripts/exam_answer.sh --student-path . --moulinette-path ./moulinette-ubuntu \
    --questions "What is PagedAttention?,How to deploy vLLM?,What models does vLLM support?"
```

It lists the questions whose sources were retrieved correctly and runs
`answer` on the three selected ones. At least 2 of 3 must be satisfactory.

### Edge cases

```bash
./exams/scripts/exam_edge_cases.sh --student-path .
```

| Test | Command | Current behaviour |
|---|---|---|
| Empty query | `search '' --k 10` | Returns the top-k chunks of an empty ranking, no traceback. |
| Gibberish | `search 'asdfghjkl zxcvbnm qwertyuiop' --k 10` | Returns the closest chunks, no traceback. |
| `k = 0` | `answer 'What is vLLM?' --k 0` | Empty context, the model still answers, no traceback. |
| Bad dataset path | `search_dataset --dataset_path /nonexistent/dataset.json --k 10` | No traceback (the missing argument is rejected first). With a valid `--save_directory` the missing file raises a `FileNotFoundError` — see limitation 3. |

The pass criterion of the script is "no Python traceback in stderr".

---

## 8. Reproducing the measurements

```bash
# 1. build the index
uv run python -m src index --max_chunk_size 2000

# 2. retrieve for the docs dataset
uv run python -m src search_dataset \
  --dataset_path data/datasets/public/UnansweredQuestions/dataset_docs_public.json \
  --k 10 --save_directory data/output/search_results

# 3. score it
./moulinette-ubuntu evaluate_student_search_results \
  data/output/search_results/dataset_docs_public.json \
  data/datasets/public/AnsweredQuestions/dataset_docs_public.json \
  --k 10 --max_context_length 2000 --threshold 0.8
```

The same three steps with `dataset_code_public.json` and `--threshold 0.5`
give the code numbers. Repeat with the `private` datasets for the official
figures.

---

## 9. Related documents

- [`architecture.md`](architecture.md) — the pipeline, the diagrams, the
  configuration and the known limitations.
- [`modules.md`](modules.md) — every file and its API.
