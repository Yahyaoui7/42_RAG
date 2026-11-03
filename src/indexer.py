from __future__ import annotations

import pickle
import chromadb
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
    FILE_METADATA,
)
from src.scanner import scan_files

from time import perf_counter


@dataclass
class IndexData:
    bm25: bm25s.BM25 | None
    chunks: list[MinimalSource]
    tokenized_corpus: list[
        list[str]
    ]  # kept for save/load round-trip and debugging


class Index:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        client = chromadb.PersistentClient(path=PROCESSED_DIR)
        self.collection = client.get_or_create_collection(
            name="semantic_index"
        )
        if Path(FILE_METADATA).exists():
            with open(FILE_METADATA, "r") as file:
                self.manifest = json.load(file)
        else:
            self.manifest = {}

    def build_lexical_index(self, chunks: list[MinimalSource]) -> IndexData:
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

    def build_semantic_index(self, chunks: list[MinimalSource]) -> np.ndarray:
        return self.model.encode(
            [chunk.text for chunk in chunks],
            show_progress_bar=True,
        )

    def save_semantic_index(
        self,
        vectors: np.ndarray,
        chunks: list[MinimalSource],
        path: Path = PROCESSED_DIR,
    ) -> None:
        batch_size = 5000

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            self.collection.upsert(
                ids=[str(chunk.id) for chunk in batch],
                documents=[chunk.text for chunk in batch],
                embeddings=vectors[i : i + batch_size].tolist(),
                metadatas=[
                    {
                        "file_path": str(chunk.file_path),
                        "first_character_index": chunk.first_character_index,
                        "last_character_index": chunk.last_character_index,
                    }
                    for chunk in batch
                ],
            )

    def save_lexical_index(
        self, index: IndexData, path: Path = INDEX_PATH
    ) -> None:
        """Pickle IndexData to path (protocol 4). Ensures parent exists."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(index, f, protocol=4)

    def load_index(self, path: Path = INDEX_PATH) -> IndexData:
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

    def _chunk_file(
        self, path: Path, max_chunk_size: int
    ) -> list[MinimalSource]:
        """Dispatch to the right langchain chunker by suffix, else generic text chunker."""
        suffix = Path(path).suffix.lower()
        if suffix == ".py":
            return chunk_python_file(path, max_chunk_size=max_chunk_size)
        elif suffix == ".md":
            return chunk_markdown_file(path, max_chunk_size=max_chunk_size)
        else:
            return chunk_text_file(path, max_chunk_size=max_chunk_size)

    def build_and_save_from_raw(
        self,
        max_chunk_size: int = MAX_CHUNK_SIZE,
    ) -> None:

        files_dict = scan_files(RAW_DIR)

        all_files: list[Path] = []
        for lst in files_dict.values():
            all_files.extend(lst)
        all_files = sorted(all_files)

        chunks: list[MinimalSource] = []
        new_chunck = []

        self.remove_removed_file(all_files)

        loop = perf_counter()
        for file_path in tqdm(all_files, desc="Chunking"):

            data = self.is_file_dirty(file_path)

            if not data:
                try:

                    file_chunks = self._chunk_file(file_path, max_chunk_size)
                    chunks.extend(file_chunks)
                    new_chunck.extend(file_chunks)

                    self.remove_modifie_file(file_path)

                    self.save_metadata_file(file_path, file_chunks)

                except Exception as e:
                    print(f"Skip {file_path}: {e}", file=sys.stderr)

            else:
                chunks.extend(
                    MinimalSource(**chunk) for chunk in data["chunks"]
                )
                ("Chunking and metadata", loop)

        self._save_manifest_file()

        lexica_index = self.build_lexical_index(chunks)

        self.save_lexical_index(lexica_index)

        semantic_index = self.build_semantic_index(new_chunck)

        self.save_semantic_index(semantic_index, new_chunck)

        print(
            f"Indexed {len(chunks)} chunks from "
            f"{len(all_files)} files -> {INDEX_PATH}"
        )

    def save_metadata_file(
        self,
        file_path: Path,
        chunks: list[MinimalSource],
    ) -> None:

        meta_data = {
            "mtime": self.get_time(file_path),
            "hash": self.calculate_hash(file_path),
            "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
        }
        self.manifest[str(file_path)] = meta_data

    def is_file_dirty(
        self, file_path: Path | str, save_path: Path = FILE_METADATA
    ) -> dict | bool:
        if not Path(save_path).exists():
            return False

        file_path = str(file_path)

        with open(save_path, "r") as file:
            manifest = json.load(file)

        if file_path not in manifest:
            return False

        metadata = manifest[file_path]
        current_mtime = self.get_time(file_path)

        if metadata["mtime"] != current_mtime:
            current_hash = self.calculate_hash(Path(file_path))

            if metadata["hash"] != current_hash:
                return False

            # Content is unchanged; only the modification time changed.
            metadata["mtime"] = current_mtime

            with open(save_path, "w") as file:
                json.dump(manifest, file, indent=4)

        return metadata

    def get_time(self, file_path: Path | str) -> float:
        return Path(file_path).stat().st_mtime

    def calculate_hash(self, file_path: Path) -> str:
        import hashlib

        sha256 = hashlib.sha256()

        with file_path.open("rb") as file:
            for block in iter(lambda: file.read(4096), b""):
                sha256.update(block)

        return sha256.hexdigest()

    def remove_modifie_file(self, file_path: Path | str) -> None:

        self.collection.delete(where={"file_path": str(file_path)})
        self.manifest.pop(str(file_path), None)

    def remove_removed_file(
        self,
        all_files_path: list[Path],
    ) -> None:

        manifest_files = set(self.manifest.keys())
        current_files = {str(path) for path in all_files_path}

        deleted_files = manifest_files - current_files

        for file_path in deleted_files:
            self.collection.delete(where={"file_path": str(file_path)})
            self.manifest.pop(str(file_path), None)

    def measure_time(self, name: str, start: float) -> None:
        elapsed = perf_counter() - start
        print(f"[TIME] {name}: {elapsed:.2f} seconds")

    def _save_manifest_file(self) -> None:
        with open(FILE_METADATA, "w") as file:
            json.dump(self.manifest, file, indent=4)
