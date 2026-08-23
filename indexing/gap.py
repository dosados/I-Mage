from pathlib import Path

from db.database import Database
from db.hash import resolved_path_key
from db.types import MODULE_CLIP, MODULE_FACES, MODULE_YOLO, ImageRecord, ModuleStatus


def module_gap_paths(
    db: Database | None,
    paths: list[Path],
    module: str,
    *,
    records_by_path: dict[str, ImageRecord] | None = None,
) -> list[Path]:
    if records_by_path is None:
        if db is None:
            raise ValueError("db is required when records_by_path is not provided")
        records_by_path = db.images.map_records_by_path(paths)

    gap: list[Path] = []
    for path in paths:
        record = records_by_path.get(resolved_path_key(path))
        if record is None:
            gap.append(path)
            continue
        module_index = record.modules.get(module)
        if module_index is None or module_index.status != ModuleStatus.DONE:
            gap.append(path)
    return gap


def clip_gap_paths(
    db: Database | None,
    paths: list[Path],
    *,
    records_by_path: dict[str, ImageRecord] | None = None,
) -> list[Path]:
    return module_gap_paths(db, paths, MODULE_CLIP, records_by_path=records_by_path)


def faces_gap_paths(
    db: Database | None,
    paths: list[Path],
    *,
    records_by_path: dict[str, ImageRecord] | None = None,
) -> list[Path]:
    return module_gap_paths(db, paths, MODULE_FACES, records_by_path=records_by_path)


def yolo_gap_paths(
    db: Database | None,
    paths: list[Path],
    *,
    records_by_path: dict[str, ImageRecord] | None = None,
) -> list[Path]:
    return module_gap_paths(db, paths, MODULE_YOLO, records_by_path=records_by_path)
