import fnmatch
from pathlib import Path

from io_utils.fs import IMAGE_SUFFIXES, collect_files


def _is_ignored(path: Path, ignore_globs: list[str]) -> bool:
    posix = path.as_posix()
    for pattern in ignore_globs:
        normalized = pattern.strip()
        if not normalized:
            continue
        if fnmatch.fnmatch(posix, normalized):
            return True
        if fnmatch.fnmatch(path.name, normalized):
            return True
    return False


def collect_scoped_files(
    include_directories: list[Path | str],
    ignore_globs: list[str] | None = None,
    *,
    recursive: bool = True,
) -> list[Path]:
    ignore = ignore_globs or []
    paths: list[Path] = []
    seen: set[Path] = set()

    for raw_directory in include_directories:
        directory = Path(raw_directory).expanduser().resolve()
        if not directory.is_dir():
            continue
        for path in collect_files(directory, IMAGE_SUFFIXES, recursive=recursive):
            resolved = path.resolve()
            if resolved in seen:
                continue
            if _is_ignored(resolved, ignore):
                continue
            seen.add(resolved)
            paths.append(resolved)

    return sorted(paths)
