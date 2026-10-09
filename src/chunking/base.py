from dataclasses import dataclass

from src.models import MinimalSource


@dataclass(frozen=True)
class Chunk:
    text: str
    file_path: (
        str  # verbatim "data/raw/..." relative to repo root, forward slashes
    )
    first_character_index: int  # inclusive start in original file content
    last_character_index: (
        int  # exclusive end: content[start:end] == text  (invariant)
    )


def split_oversized_chunk(chunk: Chunk, max_chunk_size: int) -> list[Chunk]:
    """Deprecated guard — kept only for import compatibility.

    Chunkers (python_chunker / markdown_chunker) already enforce
    ``len(text) <= max_chunk_size`` while building chunks, so this
    function must not do blind ``range(0, len, max)`` slicing which
    breaks Python (mid-def / mid-string) and Markdown (mid-header /
    mid-fence) structure.

    If ``len(chunk.text) <= max_chunk_size`` returns ``[chunk]``.
    Otherwise raises ``ValueError`` so a bug in a chunker is caught
    loudly at ``index`` time instead of being silently masked.
    """

    if len(chunk.text) <= max_chunk_size:
        return [chunk]

    raise ValueError(
        f"chunker emitted oversized chunk {len(chunk.text)} > {max_chunk_size} "
        f"for {chunk.file_path} [{chunk.first_character_index}:"
        f"{chunk.last_character_index}]. "
        f"Fix python_chunker/markdown_chunker to split that logical unit "
        f"internally (line-aware for .py, paragraph-aware for .md)."
    )


def validate_chunk(chunk: Chunk, file_content: str) -> bool:
    """Return True iff invariants hold.

    - ``last - first == len(text)``
    - ``len(text) <= 2000``
    - ``file_content[first:last] == text``
    - indices within bounds
    """

    if chunk.last_character_index - chunk.first_character_index != len(chunk.text):
        return False
    if len(chunk.text) > 2000:
        return False
    if chunk.first_character_index < 0 or chunk.last_character_index > len(
        file_content
    ):
        return False
    if (
        file_content[chunk.first_character_index : chunk.last_character_index]
        != chunk.text
    ):
        return False
    return True


def chunk_to_minimal_source(chunk: Chunk) -> MinimalSource:
    """Convert Chunk -> MinimalSource (helper for retriever)."""

    return MinimalSource(
        file_path=chunk.file_path,
        first_character_index=chunk.first_character_index,
        last_character_index=chunk.last_character_index,
    )
