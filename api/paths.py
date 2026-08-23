from pathlib import Path

from io_utils import IMAGE_SUFFIXES, collect_files
from io_utils.scan import collect_scoped_files


def resolve_image_paths(
    directory: Path,
    *,
    limit: int | None = None,
    recursive: bool = True,
) -> list[Path]:
    paths = collect_files(directory, IMAGE_SUFFIXES, recursive=recursive)
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"no images found in {directory}")
    return paths


def resolve_scoped_image_paths(
    include_directories: list[str],
    ignore_globs: list[str] | None = None,
    *,
    limit: int | None = None,
) -> list[Path]:
    paths = collect_scoped_files(include_directories, ignore_globs)
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise ValueError("no images found in configured scan directories")
    return paths
