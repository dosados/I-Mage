import hashlib
from pathlib import Path


def compute_content_hash(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_file_stat(path: Path) -> tuple[float, int]:
    stat = path.stat()
    return stat.st_mtime, stat.st_size


def resolved_path_key(path: Path | str) -> str:
    """Canonical string for the ``images.path`` column.

    Absolute paths are kept as-is (``collect_scope_paths`` already resolves them).
    Relative paths call ``Path.resolve()`` once. Avoids tens of thousands of
    redundant realpath syscalls during reconcile / gap / status.
    """
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(p.resolve())
