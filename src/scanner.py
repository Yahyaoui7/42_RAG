from pathlib import Path
from src.config import RAW_DIR


def scan_files(root: Path = RAW_DIR) -> dict[str, list[Path]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")

    files = {
        "py": sorted(root.rglob("*.py")),
        "md": sorted(root.rglob("*.md")),
        "txt": sorted(root.rglob("*.txt")),
    }

    return files



if __name__ == "__main__":
    files = scan_files()
    print(files["txt"][0])
