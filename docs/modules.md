# Module reference

Every file of the project, what it is responsible for, what it exposes, and the
details that are easy to get wrong when reading or modifying the code.

---

## Quick reference

| File | Responsibility |
|---|---|
| `src/__main__.py` | Fire CLI: the 6 commands required by the subject. |
| `src/config.py` | Single source of truth for paths, sizes and model settings. |
| `src/models.py` | The pydantic models exchanged between the stages. |
| `src/scanner.py` | Discovers the corpus files to index. |
| `src/chunking/base.py` | The `Chunk` type, the invariant checker, the `Chunk` → `MinimalSource` conversion. |
| `src/chunking/python_chunker.py` | Python code chunking strategy. |
| `src/chunking/markdown_chunker.py` | Markdown chunking strategy. |
| `src/chunking/text_chunker.py` | Generic text / other-extension chunking strategy. |
| `src/chunking/__init__.py` | Re-exports the chunking public API. |
| `src/indexer.py` | Builds, saves and loads the BM25 and semantic indexes. |
| `src/retriever.py` | BM25 search and dataset-level formatting. |
| `src/prompts.py` | Context budget and prompt assembly. |
| `src/generate.py` | Qwen3-0.6B answer generation. |
| `src/evaluator.py` | Local recall@k (for iteration only, never for grading). |
| `constants.py` | Path constants at the repository root (not used by `src/`). |
| `Makefile` | `install`, `run`, `debug`, `clean`, `lint`. |
| `exams/scripts/` | The grader scripts that drive the pipeline during the defence. |
| `evaluations/` | Recorded logs of the retrieval, answer and edge-case runs. |

---

## `src/__main__.py` — the CLI

Fire entrypoint. `python -m src <command>` dispatches to a method of the `CLI`
class; every argument becomes a flag (`--dataset_path`, or `-d` as used in the
Makefile).

| Command | Arguments (default) | Does |
|---|---|---|
| `index` | `--max_chunk_size` (2000) | Scans, chunks, builds and persists the index. |
| `search` | `query`, `--k` (10) | Prints the top-k source locations for one question. |
| `search_dataset` | `--dataset_path`, `--save_directory`, `--k` (10) | Runs the search over a dataset and writes a `StudentSearchResults` JSON. |
| `answer` | `query`, `--k` | Answers one question and prints the answer. |
| `answer_dataset` | `--student_search_results_path`, `--save_directory` | Answers a whole search-results file and writes a `StudentSearchResultsAndAnswer` JSON. |
| `evaluate` | `--student_search_results_path`, `--dataset_path` | Prints the local recall@k. |

Notes:

- The defaults come from `src/config.py`, so the commands can be run with no
  argument at all; every path is still overridable from the command line.
- `index` is the only command with its own `try/except`; it prints the error
  and exits with status 1. The other commands let exceptions propagate — see
  limitation 3 in [`architecture.md`](architecture.md#11-known-limitations).
- `search_dataset` creates the output directory if needed and writes the file
  under the **same name as the input dataset**, e.g.
  `data/output/search_results/dataset_docs_public.json`.
- The `Searcher` is instantiated once per command and reused for every
  question of the dataset — the index is loaded once, not once per question.

---

## `src/config.py` — configuration

Pure constants, no logic. Everything the rest of the code needs to know about
paths, sizes and the model. See the
[configuration table](architecture.md#10-configuration) for the full list.

- `RAW_DIR` / `PROCESSED_DIR` / `INDEX_PATH` — corpus root and index locations.
- `MAX_CHUNK_SIZE` (2000) — the hard limit imposed by the grader's
  `--max_context_length`.
- `TOP_K` (10) — default number of sources returned.
- Dataset and output paths, used as the defaults of the CLI commands.
- `QWEN_MODEL_ID` / `QWEN_MAX_NEW_TOKENS` / `QWEN_TEMPERATURE` — generation
  settings.

---

## `src/models.py` — data models

The pydantic models that carry data between stages. They are the contract of
the project: the dataset loader, the retriever, the generator and the output
files all speak these types.

| Model | Role |
|---|---|
| `MinimalSource` | One source location: `file_path`, `first_character_index`, `last_character_index` and the optional chunk `text`. |
| `UnansweredQuestion` | A question without an answer; `question_id` defaults to a UUID. |
| `AnsweredQuestion` | A question with its reference `sources` and `answer`. |
| `RagDataset` | A dataset file: `rag_questions: List[AnsweredQuestion \| UnansweredQuestion]`. |
| `MinimalSearchResults` | One question with its `retrieved_sources`. |
| `MinimalAnswer` | One question with its `retrieved_sources` and the generated `answer`. |
| `StudentSearchResults` | The output of `search_dataset`. |
| `StudentSearchResultsAndAnswer` | The output of `answer_dataset`. |

Notes:

- `MinimalSource.text` defaults to `""`, so it is optional for consumers. It
  is an addition to the model required by the subject, and the `moulinette`
  accepts it (`Student data is valid: True`).
- `AnsweredQuestion` and `UnansweredQuestion` are siblings, not parent/child
  of a common base beyond `UnansweredQuestion`; the union in `RagDataset`
  makes the dataset loader accept both shapes.

---

## `src/scanner.py` — corpus discovery

```python
scan_files(root: Path = RAW_DIR) -> dict[str, list[Path]]
```

Walks the corpus recursively and returns the files grouped by extension:

```python
{"py": [...], "md": [...], "txt": [...]}
```

- The result is sorted, so indexing is deterministic.
- Only `.py`, `.md` and `.txt` are collected; the other 400+ extensions of the
  vLLM repository (images, `.cu`, `.json`, `.jinja`, …) are intentionally left
  out — they hold no answerable prose or code for this corpus.
- The current corpus yields **1952 files**.

---

## `src/chunking/base.py` — the shared chunk type

- `Chunk` — frozen dataclass with `text`, `file_path`, `first_character_index`,
  `last_character_index`. The comments in the file state the three invariants
  (span matches the text length, the text is a verbatim slice of the file, and
  the chunk is never wider than the limit).
- `validate_chunk(chunk, file_content, max_chunk_size) -> bool` — checks the
  invariants against the real file content. **Not called by the indexer
  today** (limitation 5), it is the natural place to add a build-time
  assertion.
- `chunk_to_minimal_source(chunk) -> MinimalSource` — the single conversion
  point from the internal chunk to the public source location.

`split_oversized_chunk` is commented out on purpose: blind slicing would cut
Python code in the middle of a definition and Markdown in the middle of a
fenced block, so the chunkers are responsible for producing small enough
chunks themselves.

---

## `src/chunking/python_chunker.py` — Python chunking

```python
chunk_python_file(file_path: Path, max_chunk_size: int = 2000) -> list[Chunk]
```

- Uses LangChain's `RecursiveCharacterTextSplitter.from_language(Language.PYTHON)`,
  which splits on Python-aware separators (indentation levels, `def`/`class`
  boundaries) before falling back to plain line and character splitting.
- `chunk_overlap=0`, `add_start_index=True`.
- The resulting chunks respect the language structure far better than a naive
  character split, which matters for code questions: the retrieved snippet is
  usually a complete function rather than half of one.
- Unreadable files and empty files return `[]` instead of raising, so one bad
  file can never abort an indexing run.

---

## `src/chunking/markdown_chunker.py` — Markdown chunking

```python
chunk_markdown_file(file_path: Path, max_chunk_size: int = 2000) -> list[Chunk]
```

- Same mechanism with `Language.MARKDOWN`: the splitter is aware of headers
  (`#`, `##`, …), horizontal rules and code fences.
- Because Markdown sections are usually self-contained, a retrieved Markdown
  chunk frequently answers the question on its own — this is what gives the
  docs dataset its 81–85 % recall@5 against 50–56 % for code.

---

## `src/chunking/text_chunker.py` — generic text chunking

```python
chunk_text_file(file_path: Path, max_chunk_size: int = 2000) -> list[Chunk]
```

- Fallback for every extension that is not `.py` or `.md` (`.txt`, and in
  practice files such as `CMakeLists.txt`).
- Uses the plain `RecursiveCharacterTextSplitter`, which splits on paragraphs
  then on lines then on characters.
- This is the third strategy that makes the `_chunk_file` dispatch in
  `src/indexer.py` total: every scanned file has exactly one strategy.

> The three chunkers share the same skeleton — read, guard, split, repair the
> offset, build the `Chunk` list — and differ only in the splitter
> configuration. The offset-repair block is duplicated in the three files; it
> is the part to keep identical across them.

---

## `src/chunking/__init__.py`

Re-exports `Chunk`, `chunk_to_minimal_source`, `validate_chunk` and the three
chunker functions, so the indexer imports everything from `src.chunking`.

---

## `src/indexer.py` — building the index

| Function | Role |
|---|---|
| `build_and_save_from_raw(max_chunk_size)` | The entry point of the `index` command: scan, chunk every file, build both indexes, save them. |
| `_chunk_file(path, max_chunk_size)` | Dispatch on the suffix to the right chunker. |
| `build_lexical_index(chunks)` | Tokenize the chunks and build the BM25 model; returns an `IndexData` with `bm25=None` if the corpus is empty. |
| `build_semantic_index(chunks)` | Encode the chunks with `all-MiniLM-L6-v2`, L2-normalized. |
| `save_lexical_index(index, path)` | Pickle the `IndexData` (protocol 4). |
| `save_semantic_index(vectors, path)` | Save the embedding matrix as `.npy` next to the pickle. |
| `load_index(path)` | Load the pickle; raises a `FileNotFoundError` with the command to run if it is missing. |

`IndexData` is the dataclass that is actually persisted:

```python
@dataclass
class IndexData:
    bm25: bm25s.BM25 | None
    chunks: list[Chunk]
    tokenized_corpus: list[list[str]]
```

Notes:

- The chunking loop is wrapped in a per-file `try/except`: a file that cannot
  be read is reported on stderr and skipped, the rest of the corpus is still
  indexed.
- The chunk order is deterministic (sorted file list), so the chunk index of
  the pickle always matches the order of `IndexData.chunks`.
- The `__main__` demo block at the bottom of the file still calls an undefined
  `tokenize()` (limitation 4) — it is dead code from an earlier design.

---

## `src/retriever.py` — search

`Searcher` is the only class that reads the index.

| Method | Role |
|---|---|
| `__init__(index_path)` | Loads the pickle once. |
| `search(query, k)` | Tokenizes the query, asks BM25 for the top-k chunks, converts them to `MinimalSource`. Returns `[]` if `k <= 0`, if the query is empty or if the index is empty. |
| `search_one(query, index_path, k)` | Same as `search`, but reloads the index when a different path is given. |
| `format_as_student_results(questions, k)` | Runs the search over a dataset (with a `tqdm` progress bar) and returns a `StudentSearchResults`. |

Notes:

- The query is tokenized with `bm25s.tokenize`, the very same call used at
  index time — this is what keeps the recall numbers reproducible.
- `bm25.retrieve(tokens, k=k)` returns the ranked chunk indices; the scores
  are not needed afterwards because the ground truth is matched on the file and
  the character range, not on the score.

---

## `src/prompts.py` — prompt assembly

| Member | Role |
|---|---|
| `MAX_CONTEXT_CHARS = 6000` | Total size of the context pasted into the prompt. |
| `load_chunk_text(chunk)` | Reads a source range from the file — **unused today**, the context comes from `MinimalSource.text`. |
| `truncate_contexts(contexts, budget)` | Adds contexts in retrieval order until the character budget is reached. |
| `build_prompt(question, contexts)` | Joins the contexts and returns the user message. |

The prompt has a fixed shape:

```text
Context:
        <retrieved context>

        Question:
        <question>
```

The instruction to the model lives in the system prompt of
`src/generate.py`, not here: this module only assembles the context.

---

## `src/generate.py` — answer generation

| Member | Role |
|---|---|
| `__init__()` | Loads the Qwen3-0.6B tokenizer and model (`device_map="auto"`) and defines the system prompt. |
| `generate_answer(prompt)` | Applies the chat template (`enable_thinking=False`), generates greedily with `max_new_tokens=300` and decodes the new tokens only. |
| `answer_dataset(student_result, save_directory)` | Answers every question of a `StudentSearchResults` file and writes a `StudentSearchResultsAndAnswer` file. |

The system prompt instructs the model to answer only from the context, to stay
concise, and to answer "I don't know." when the context does not contain the
answer — the standard anti-hallucination guard for a small model.

> `answer_dataset` currently fails on its last line (limitation 1): the output
> path is built from `Path(student_result)`, where `student_result` is the
> pydantic model instead of the input file path.

---

## `src/evaluator.py` — local recall

A local approximation of the official metric, used while iterating. The
official numbers always come from the `moulinette`.

| Method | Role |
|---|---|
| `__init__(student_search_results_path, dataset_path)` | Loads both JSON files and indexes the student sources by `question_id`. |
| `check_is_valid(raw_source, student_source)` | `True` when the paths are identical and the ranges overlap with an IoU of at least 0.05. |
| `check_top_k(raw_source, student_sources, k)` | `True` when a matching source is found in the first k results. |
| `get_score()` | Recall@1, @3, @5 and @10 over the dataset. |

The IoU threshold and the "same file" rule are the same as the official ones.
The difference: only the **first** reference source of each question is scored
(limitation 7).

---

## `constants.py` (repository root)

Path constants that mirror `src/config.py`. Nothing in `src/` imports this
file — `src/config.py` is the one that is actually used. Keeping both in sync
is manual work with no benefit; this file is a leftover of an earlier layout.

---

## `Makefile`

| Target | Action |
|---|---|
| `install` | `pip install uv` then `uv sync`. |
| `run` | Full pipeline: `index`, then `search_dataset` + `evaluate` on the docs dataset, then the same on the code dataset. |
| `debug` | `python -m pdb -m src`. |
| `clean` | `pyclean` + removal of `.mypy_cache`. |
| `lint` | `flake8 .` and `mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs`. |

> The `run` target uses the short Fire aliases (`-d`, `-s`), which Fire
> accepts. It currently fails on the lint findings and on
> `answer_dataset` (see [`architecture.md`](architecture.md#11-known-limitations)).

---

## `exams/scripts/` and `evaluations/`

- `exams/scripts/exam_retrieval.sh` — drives `index`, `search_dataset` and
  the `moulinette`, and checks the 300 s / 90 s / 80 % / 50 % budgets.
- `exams/scripts/exam_answer.sh` — asks three questions and prints the answers
  for a human judgement.
- `exams/scripts/exam_edge_cases.sh` — runs the four degenerate inputs and
  fails on any Python traceback.
- `evaluations/<kind>/<timestamp>/` — the recorded stdout, stderr and summary
  of each run. The retrieval summaries are the fastest way to see whether the
  last change improved or degraded the recall.
