from src.config import MAX_CHUNK_SIZE
from src.models import MinimalSource


def validate_chunk(
    chunk: MinimalSource,
    file_content: str,
    max_chunk_size: int = MAX_CHUNK_SIZE,
) -> bool:
    """Return True iff chunk invariants hold.

    - last - first == len(text)
    - len(text) <= max_chunk_size
    - file_content[first:last] == text
    - indices within bounds
    """
    span = chunk.last_character_index - chunk.first_character_index
    if span != len(chunk.text):
        return False
    if len(chunk.text) > max_chunk_size:
        return False
    if chunk.first_character_index < 0 or chunk.last_character_index > len(
        file_content
    ):
        return False
    if (
        file_content[chunk.first_character_index:chunk.last_character_index]
        != chunk.text
    ):
        return False
    return True
