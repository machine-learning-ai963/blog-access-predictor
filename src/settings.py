from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

DEFAULT_SITE_ROOT = "https://ai-fukushi.net"
DEFAULT_ARCHIVE_URL = "https://ai-fukushi.net/archive/"

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
MAX_EMBEDDING_BYTES = 24000
RANDOM_STATE = 42

DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "model_bundle.joblib"
