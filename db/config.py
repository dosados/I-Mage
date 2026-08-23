import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "data" / "i-mage.db"

SCHEMA_VERSION = 5


def resolve_db_path(path: Path | str | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()
    return Path(os.environ.get("IMAGE_DB_PATH", DEFAULT_DB_PATH)).expanduser().resolve()
