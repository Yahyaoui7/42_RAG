from pathlib import Path

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from src.chunking.base import Chunk
from src.config import DEFAULT_MAX_CHUNK_SIZE


def chunk_markdown_file(
    file_path: Path, max_chunk_size: int = 2000
) -> list[Chunk]:

    file_path = Path(file_path)
    file_path_str = str(file_path)

    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except (OSError, UnicodeError):
        return []

    if not content.strip():
        return []

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=max_chunk_size,
        chunk_overlap=0,
        add_start_index=True,
    )

    documents = splitter.create_documents([content])
    chunks: list[Chunk] = []
    for doc in documents:
        text: str = doc.page_content
        if not text.strip():
            continue
        # add_start_index puts char offset in metadata
        start: int = doc.metadata["start_index"]

        end = start + len(text)
        if content[start:end] != text:
            found = content.find(text, start)
            if found != -1:
                start = found
                end = start + len(text)
        chunks.append(
            Chunk(
                text=text,
                file_path=file_path_str,
                first_character_index=start,
                last_character_index=end,
            )
        )
    return chunks


if __name__ == "__main__":
    test_path = Path("data/raw/CODE_OF_CONDUCT.md")
    if test_path.exists():
        chunks = chunk_markdown_file(test_path)
        for chunk in chunks:
            print("\n\n ---------------- ")
            print("start: ", chunk.first_character_index)
            print("end: ", chunk.last_character_index, "\n")
            print(chunk.text)
    else:
        print(f"Test file not found: {test_path}")
        # synthetic quick test
