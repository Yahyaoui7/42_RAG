from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import bm25s
from tqdm import tqdm

from src.chunking.base import Chunk
from src.chunking.markdown_chunker import chunk_markdown_file
from src.chunking.python_chunker import chunk_python_file
from src.chunking.text_chunker import chunk_text_file
from src.config import (
    BM25_B,
    BM25_K1,
    DEFAULT_MAX_CHUNK_SIZE,
    INDEX_PATH,
    RAW_DIR,
)
from src.scanner import scan_files
from src.tokenizer import tokenize


@dataclass
class IndexData:
    bm25: bm25s.BM25 | None
    chunks: list[Chunk]
    tokenized_corpus: list[
        list[str]
    ]  # kept for save/load round-trip and debugging


def build_index(
    chunks: list[Chunk], k1: float = BM25_K1, b: float = BM25_B
) -> IndexData:
    """Tokenize all chunks with our regex tokenizer (keeps not/none), build bm25s index.

    Empty chunks -> IndexData with bm25=None (valid, not crash).
    We feed pre-tokenized list[list[str]] to bm25s.BM25.index() so bm25s.tokenizer
    (which drops stopwords like 'not') is never called.
    """
    tokenized_corpus: list[list[str]] = [
        tokenize(chunk.text) for chunk in chunks
    ]
    if not tokenized_corpus:
        return IndexData(bm25=None, chunks=chunks, tokenized_corpus=[])

    retriever = bm25s.BM25()
    retriever.index(tokenized_corpus, show_progress=False)
    return IndexData(
        bm25=retriever, chunks=chunks, tokenized_corpus=tokenized_corpus
    )


def save_index(index: IndexData, path: Path = INDEX_PATH) -> None:
    """Pickle IndexData to path (protocol 4). Ensures parent exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(index, f, protocol=4)


def load_index(path: Path = INDEX_PATH) -> IndexData:
    """Load IndexData from pickle. Raises FileNotFoundError with clear message if missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Index not found at {path}. Run `index --max_chunk_size 2000` first."
        )
    with path.open("rb") as f:
        return pickle.load(f)


def _chunk_file(path: Path, max_chunk_size: int) -> list[Chunk]:
    """Dispatch to the right langchain chunker by suffix, else generic text chunker."""
    suffix = Path(path).suffix.lower()
    if suffix == ".py":
        return chunk_python_file(path, max_chunk_size=max_chunk_size)
    elif suffix == ".md":
        return chunk_markdown_file(path, max_chunk_size=max_chunk_size)
    else:
        return chunk_text_file(path, max_chunk_size=max_chunk_size)


def build_and_save_from_raw(
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> None:
    """Scan data/raw/, chunk every file via langchain, build BM25, save to INDEX_PATH.

    Per-file try/except — one bad file never aborts the whole index.
    Asserts len(c.text) <= min(max_chunk_size, 2000) before save (moulinette limit).
    """
    files_dict = scan_files(RAW_DIR)
    all_files: list[Path] = []
    for lst in files_dict.values():
        all_files.extend(lst)
    all_files = sorted(all_files)

    chunks: list[Chunk] = []
    for file_path in tqdm(all_files, desc="Chunking"):
        try:
            file_chunks = _chunk_file(file_path, max_chunk_size)
            chunks.extend(file_chunks)
        except Exception as e:
            print(f"Skip {file_path}: {e}", file=sys.stderr)

    index = build_index(chunks)
    save_index(index)
    print(
        f"Indexed {len(chunks)} chunks from {len(all_files)} files -> {INDEX_PATH}"
    )


if __name__ == "__main__":
    # Demo: what build_index does step by step
    # Run with: uv run python -m src.indexer
    print("=== 1. build_index keeps not/none (your example) ===")
    demo_chunks = [
        Chunk(
            text="if not user.is_active: return None",
            file_path="data/raw/a.py",
            first_character_index=0,
            last_character_index=33,
        ),
        Chunk(
            text="if user.is_active: return user",
            file_path="data/raw/b.py",
            first_character_index=0,
            last_character_index=30,
        ),
    ]
    print(f"Input chunks: {[c.text for c in demo_chunks]}")
    demo_index = build_index(demo_chunks)
    print(f"Our tokenize keeps 'not': {demo_index.tokenized_corpus}")
    q = "not active"
    scores = demo_index.bm25.get_scores(tokenize(q))
    print(f"Query '{q}' tokens: {tokenize(q)}")
    print(
        f"BM25 scores: {scores}  (doc 0 with 'not' scores higher: {scores[0]:.3f} > {scores[1]:.3f})"
    )

    # print("\n=== 2. Chunk distribution by file type ===")
    # files_dict = scan_files(RAW_DIR)
    # print({k: len(v) for k, v in files_dict.items()}, "total", sum(len(v) for v in files_dict.values()))

    # print("\n=== 3. Full build from data/raw/ (14458 chunks expected) ===")
    # print("Running build_and_save_from_raw(2000)...")
    # build_and_save_from_raw(DEFAULT_MAX_CHUNK_SIZE)

    # print("\n=== 4. Verify saved index ===")
    # loaded = load_index()
    # print(f"Loaded {len(loaded.chunks)} chunks, max len {max(len(c.text) for c in loaded.chunks)}")
    # print(f"All <=2000: {all(len(c.text) <= 2000 for c in loaded.chunks)}")
    # print(f"All <=DEFAULT_MAX_CHUNK_SIZE: {all(len(c.text) <= DEFAULT_MAX_CHUNK_SIZE for c in loaded.chunks)}")
    # q2 = "vllm attention"
    # scores2 = loaded.bm25.get_scores(tokenize(q2))
    # top = int(scores2.argmax())
    # print(f"Test query '{q2}' tokens: {tokenize(q2)}")
    # print(f"Top hit [{top}] score {scores2[top]:.3f}: {loaded.chunks[top].file_path}")
    # print(f"Text preview: {loaded.chunks[top].text[:200]!r}...")
    # print(f"Index file size: {INDEX_PATH.stat().st_size / 1e6:.2f} MB")
    # print("\nDone. Delete with: rm data/processed/index.pkl")
