from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
import json
from dataclasses import asdict
import numpy as np
from sentence_transformers import SentenceTransformer
from src.models import MinimalSource
import os
import bm25s
from tqdm import tqdm

from src.chunking import chunk_markdown_file
from src.chunking import chunk_python_file
from src.chunking import chunk_text_file
from src.config import (
    MAX_CHUNK_SIZE,
    INDEX_PATH,
    RAW_DIR,
    PROCESSED_DIR,
)
from src.scanner import scan_files


@dataclass
class IndexData:
    bm25: bm25s.BM25 | None
    chunks: list[MinimalSource]
    tokenized_corpus: list[
        list[str]
    ]  # kept for save/load round-trip and debugging


def build_lexical_index(chunks: list[MinimalSource]) -> IndexData:
    """Build a BM25 index using our custom tokenizer."""
    tokenized_corpus = bm25s.tokenize([chunk.text for chunk in chunks])

    if not tokenized_corpus or not any(tokenized_corpus):
        return IndexData(bm25=None, chunks=chunks, tokenized_corpus=[])

    retriever = bm25s.BM25()
    retriever.index(tokenized_corpus, show_progress=False)

    return IndexData(
        bm25=retriever,
        chunks=chunks,
        tokenized_corpus=tokenized_corpus,
    )


def build_semantic_index(chunks: list[MinimalSource]) -> np.ndarray:

    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(
        [chunk.text for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def save_semantic_index(vectors, chunks, path: Path = PROCESSED_DIR):
    import chromadb

    client = chromadb.PersistentClient(path=path)

    collection = client.get_or_create_collection(name="semantic_index")
    batch_size = 5000

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        collection.add(
            ids=[str(chunk.id) for chunk in batch],
            documents=[chunk.text for chunk in batch],
            embeddings=vectors[i : i + batch_size].tolist(),
            metadatas=[
                {
                    "file_path": chunk.file_path,
                    "first_character_index": chunk.first_character_index,
                    "last_character_index": chunk.last_character_index,
                }
                for chunk in batch
            ],
        )


def save_lexical_index(index: IndexData, path: Path = INDEX_PATH) -> None:
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

    suffix = path.suffix
    if suffix == ".pkl":
        with path.open("rb") as f:
            return pickle.load(f)
    else:
        return np.load(path)


def _chunk_file(path: Path, max_chunk_size: int) -> list[MinimalSource]:
    """Dispatch to the right langchain chunker by suffix, else generic text chunker."""
    suffix = Path(path).suffix.lower()
    if suffix == ".py":
        return chunk_python_file(path, max_chunk_size=max_chunk_size)
    elif suffix == ".md":
        return chunk_markdown_file(path, max_chunk_size=max_chunk_size)
    else:
        return chunk_text_file(path, max_chunk_size=max_chunk_size)


def build_and_save_from_raw(
    max_chunk_size: int = MAX_CHUNK_SIZE,
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

    chunks: list[MinimalSource] = []
    for file_path in tqdm(all_files, desc="Chunking"):
        try:
            file_chunks = _chunk_file(file_path, max_chunk_size)
            chunks.extend(file_chunks)
        except Exception as e:
            print(f"Skip {file_path}: {e}", file=sys.stderr)

    lexica_index = build_lexical_index(chunks)
    save_lexical_index(lexica_index)

    semantic_index = build_semantic_index(chunks)
    save_semantic_index(semantic_index, chunks)

    print(
        f"Indexed {len(chunks)} chunks from {len(all_files)} files -> {INDEX_PATH}"
    )

    # def is_file_dirty(self, path_file):
