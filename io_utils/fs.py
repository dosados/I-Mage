import logging
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"})


def _normalize_suffix(suffix: str) -> str:
    lowered = suffix.lower()
    return lowered if lowered.startswith(".") else f".{lowered}"


def _normalize_suffixes(suffixes: Iterable[str]) -> frozenset[str]:
    return frozenset(_normalize_suffix(suffix) for suffix in suffixes)


def filter_image_paths(
    paths: Iterable[str | Path],
    suffixes: Iterable[str] | None = None,
) -> list[Path]:
    normalized_suffixes = _normalize_suffixes(suffixes or IMAGE_SUFFIXES)
    valid_paths: list[Path] = []

    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            logger.warning("skipping missing or non-file path: %s", path)
            continue
        if path.suffix.lower() not in normalized_suffixes:
            logger.warning("skipping unsupported image suffix: %s", path)
            continue
        valid_paths.append(path)

    return sorted(valid_paths)


def collect_files(
    root: Path,
    suffixes: Iterable[str],
    *,
    recursive: bool = True,
) -> list[Path]:
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")

    normalized_suffixes = _normalize_suffixes(suffixes)
    iterator = root.rglob("*") if recursive else root.glob("*")

    paths: list[Path] = []
    for path in iterator:
        if path.is_file() and path.suffix.lower() in normalized_suffixes:
            paths.append(path)

    return sorted(paths)
