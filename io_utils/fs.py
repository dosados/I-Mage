from collections.abc import Iterable
from pathlib import Path

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"})


def _normalize_suffix(suffix: str) -> str:
    lowered = suffix.lower()
    return lowered if lowered.startswith(".") else f".{lowered}"


def collect_files(
    root: Path,
    suffixes: Iterable[str],
    *,
    recursive: bool = True,
) -> list[Path]:
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")

    normalized_suffixes = {_normalize_suffix(suffix) for suffix in suffixes}
    iterator = root.rglob("*") if recursive else root.glob("*")

    paths: list[Path] = []
    for path in iterator:
        if path.is_file() and path.suffix.lower() in normalized_suffixes:
            paths.append(path)

    return sorted(paths)
