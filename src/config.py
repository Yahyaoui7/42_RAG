from pathlib import Path

RAW_DIR: Path = Path("data/raw")
PROCESSED_DIR: Path = Path("data/processed")
INDEX_PATH: Path = PROCESSED_DIR / "index.pkl"

DEFAULT_MAX_CHUNK_SIZE: int = 2000  # PDF VI.1 default, never emit > this
BM25_K1: float = 1.5  # term saturation, tweak if recall low
BM25_B: float = 0.75  # length norm, tweak if recall low

QWEN_MODEL_ID: str = "Qwen/Qwen3-0.6B"
QWEN_MAX_NEW_TOKENS: int = 512
QWEN_TEMPERATURE: float = 0.0  # deterministic
