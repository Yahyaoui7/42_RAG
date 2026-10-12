from __future__ import annotations
import fire, sys
from src.indexer import build_and_save_from_raw
from src.config import DEFAULT_MAX_CHUNK_SIZE


class CLI:

    def index(self, max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE) -> None:
        try:
            build_and_save_from_raw(max_chunk_size)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


def main() -> None:
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
