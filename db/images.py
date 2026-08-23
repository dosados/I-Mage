from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, joinedload

from db.hash import compute_content_hash, read_file_stat, resolved_path_key
from db.models import Detection, Face, Image, ImageClip, ImageFaces, ImageYolo, utc_now_iso
from db.types import (
    MODULE_CLIP,
    MODULE_FACES,
    MODULE_YOLO,
    ImageRecord,
    ModuleStatus,
    ReconcileResult,
)

if TYPE_CHECKING:
    from vectors.store import VectorStore

_PATH_BATCH_SIZE = 900
# Above this many paths, a full-table scan + set filter beats huge IN(...) batches
# (and avoids tens of thousands of Path.resolve() / realpath calls).
_FULL_SCAN_PATH_THRESHOLD = 1000

_MODULE_TABLES = {
    MODULE_YOLO: ImageYolo,
    MODULE_CLIP: ImageClip,
    MODULE_FACES: ImageFaces,
}


class ImageService:
    def __init__(self, session: Session, *, vector_store: VectorStore | None = None) -> None:
        self._session = session
        self._vector_store = vector_store

    def get_by_id(self, image_id: str) -> ImageRecord | None:
        image = self._session.get(Image, image_id)
        return image.to_record() if image is not None else None

    def get_by_path(self, path: Path | str) -> ImageRecord | None:
        resolved = resolved_path_key(path)
        image = self._session.scalar(select(Image).where(Image.path == resolved))
        return image.to_record() if image is not None else None

    def list_all(self, *, limit: int | None = None) -> list[ImageRecord]:
        stmt = select(Image).order_by(Image.path)
        if limit is not None:
            stmt = stmt.limit(limit)
        return [image.to_record() for image in self._session.scalars(stmt)]

    def map_records_by_path(self, paths: list[Path]) -> dict[str, ImageRecord]:
        if not paths:
            return {}

        resolved = [resolved_path_key(path) for path in paths]
        records: dict[str, ImageRecord] = {}

        for offset in range(0, len(resolved), _PATH_BATCH_SIZE):
            chunk = resolved[offset : offset + _PATH_BATCH_SIZE]
            stmt = (
                select(Image)
                .where(Image.path.in_(chunk))
                .options(
                    joinedload(Image.yolo_index),
                    joinedload(Image.faces_index),
                    joinedload(Image.clip_index),
                )
            )
            for image in self._session.scalars(stmt).unique():
                records[image.path] = image.to_record()

        return records

    def count_all(self) -> int:
        return int(self._session.scalar(select(func.count()).select_from(Image)) or 0)

    def id_to_path_for_scope(self, paths: list[Path]) -> dict[str, Path]:
        """Map catalog ``image_id -> Path`` for the given scope.

        Avoids the N+1 ``get_by_path`` loop that made every search walk the
        whole SQLite catalog one row at a time.
        """
        if not paths:
            return {}

        resolved = {resolved_path_key(path) for path in paths}
        result: dict[str, Path] = {}

        if len(resolved) >= _FULL_SCAN_PATH_THRESHOLD:
            stmt = select(Image.id, Image.path)
            for image_id, path in self._session.execute(stmt):
                if path in resolved:
                    result[image_id] = Path(path)
            return result

        path_list = list(resolved)
        for offset in range(0, len(path_list), _PATH_BATCH_SIZE):
            chunk = path_list[offset : offset + _PATH_BATCH_SIZE]
            stmt = select(Image.id, Image.path).where(Image.path.in_(chunk))
            for image_id, path in self._session.execute(stmt):
                result[image_id] = Path(path)
        return result

    def count_module_done(self, module: str) -> int:
        """Catalog-wide DONE count for ``module`` (one indexed COUNT, no path list)."""
        module_table = _MODULE_TABLES.get(module)
        if module_table is None:
            return 0
        return int(
            self._session.scalar(
                select(func.count())
                .select_from(module_table)
                .where(module_table.status == ModuleStatus.DONE.value)
            )
            or 0
        )

    def catalog_module_stats(self) -> dict[str, dict[str, int]]:
        """Fast per-module {done, total} from SQL COUNTs — no filesystem walk."""
        total = self.count_all()
        return {
            module: {"done": self.count_module_done(module), "total": total}
            for module in (MODULE_YOLO, MODULE_CLIP, MODULE_FACES)
        }

    def stat_map_by_path(
        self, paths: Iterable[Path]
    ) -> dict[str, tuple[float, int, str]]:
        """path -> (mtime, size, content_hash) without loading module relations.

        Reconcile only needs file stats to decide what changed; pulling the full
        ORM graph with joinedload(yolo/faces/clip) for the whole catalog is what
        made reconcile take minutes. This is a flat, index-friendly read.
        """
        resolved_set = {resolved_path_key(p) for p in paths}
        if not resolved_set:
            return {}

        result: dict[str, tuple[float, int, str]] = {}
        if len(resolved_set) >= _FULL_SCAN_PATH_THRESHOLD:
            stmt = select(Image.path, Image.mtime, Image.size, Image.content_hash)
            for path, mtime, size, content_hash in self._session.execute(stmt):
                if path in resolved_set:
                    result[path] = (mtime, size, content_hash)
            return result

        resolved = list(resolved_set)
        for offset in range(0, len(resolved), _PATH_BATCH_SIZE):
            chunk = resolved[offset : offset + _PATH_BATCH_SIZE]
            stmt = select(Image.path, Image.mtime, Image.size, Image.content_hash).where(
                Image.path.in_(chunk)
            )
            for path, mtime, size, content_hash in self._session.execute(stmt):
                result[path] = (mtime, size, content_hash)
        return result

    def done_paths_for_module(self, module: str, paths: list[Path]) -> set[str]:
        """Resolved path strings whose ``module`` status is DONE.

        Lets gap detection skip loading the full record graph for every image:
        gap = scope paths not in this set.
        """
        module_table = _MODULE_TABLES.get(module)
        if module_table is None or not paths:
            return set()

        resolved_set = {resolved_path_key(p) for p in paths}
        if len(resolved_set) >= _FULL_SCAN_PATH_THRESHOLD:
            stmt = (
                select(Image.path)
                .join(module_table, module_table.image_id == Image.id)
                .where(module_table.status == ModuleStatus.DONE.value)
            )
            return {path for path in self._session.scalars(stmt) if path in resolved_set}

        done: set[str] = set()
        resolved = list(resolved_set)
        for offset in range(0, len(resolved), _PATH_BATCH_SIZE):
            chunk = resolved[offset : offset + _PATH_BATCH_SIZE]
            stmt = (
                select(Image.path)
                .join(module_table, module_table.image_id == Image.id)
                .where(
                    Image.path.in_(chunk),
                    module_table.status == ModuleStatus.DONE.value,
                )
            )
            done.update(self._session.scalars(stmt))
        return done

    def count_module_done_in_paths(self, paths: list[Path], module: str) -> int:
        if not paths:
            return 0

        module_table = _MODULE_TABLES.get(module)
        if module_table is None:
            return 0

        resolved_set = {resolved_path_key(path) for path in paths}
        if len(resolved_set) >= _FULL_SCAN_PATH_THRESHOLD:
            # Prefer catalog count when the scope is essentially the whole DB.
            catalog_total = self.count_all()
            if len(resolved_set) >= catalog_total and catalog_total > 0:
                return self.count_module_done(module)
            done_paths = self.done_paths_for_module(module, paths)
            return len(done_paths)

        resolved = list(resolved_set)
        done = 0
        for offset in range(0, len(resolved), _PATH_BATCH_SIZE):
            chunk = resolved[offset : offset + _PATH_BATCH_SIZE]
            stmt = (
                select(func.count())
                .select_from(Image)
                .join(module_table, module_table.image_id == Image.id)
                .where(
                    Image.path.in_(chunk),
                    module_table.status == ModuleStatus.DONE.value,
                )
            )
            done += int(self._session.scalar(stmt) or 0)
        return done

    def upsert_from_file(
        self,
        path: Path,
        *,
        content_hash: str,
        mtime: float,
        size: int,
    ) -> ImageRecord:
        resolved_str = resolved_path_key(path)
        now = utc_now_iso()
        image = self._session.scalar(select(Image).where(Image.path == resolved_str))

        if image is None:
            image = Image(
                id=str(uuid.uuid4()),
                path=resolved_str,
                content_hash=content_hash,
                mtime=mtime,
                size=size,
                created_at=now,
                updated_at=now,
            )
            self._session.add(image)
            self._session.flush()
            return image.to_record()

        content_changed = image.content_hash != content_hash
        mtime_changed = image.mtime != mtime or image.size != size

        if content_changed:
            image.path = resolved_str
            image.content_hash = content_hash
            image.mtime = mtime
            image.size = size
            image.updated_at = now
            self._invalidate_indexed_data(image.id)
        elif mtime_changed:
            image.path = resolved_str
            image.mtime = mtime
            image.size = size
            image.updated_at = now

        self._session.flush()
        return image.to_record()

    def delete_by_id(self, image_id: str) -> None:
        image = self._session.get(Image, image_id)
        if image is not None:
            self._session.delete(image)
            self._session.flush()

    def reconcile_paths(
        self,
        paths: set[Path],
        *,
        remove_missing: bool = False,
        on_progress: Callable[[int, int], None] | None = None,
        commit_batch_size: int = 250,
    ) -> ReconcileResult:
        # Preserve Path objects keyed by DB string; avoid a second realpath pass.
        by_key: dict[str, Path] = {}
        for path in paths:
            key = resolved_path_key(path)
            by_key[key] = Path(key)
        resolved_keys = sorted(by_key)
        total = len(resolved_keys)
        # Flat (mtime, size, hash) lookup — no ORM graph / joins for the catalog.
        existing_map = self.stat_map_by_path(by_key.values())
        upserted = 0

        if on_progress is not None and total > 0:
            on_progress(0, total)

        for index, resolved_str in enumerate(resolved_keys):
            path = by_key[resolved_str]
            mtime, size = read_file_stat(path)
            existing = existing_map.get(resolved_str)

            if existing is None:
                content_hash = compute_content_hash(path)
                self.upsert_from_file(
                    path,
                    content_hash=content_hash,
                    mtime=mtime,
                    size=size,
                )
                upserted += 1
            elif existing[0] != mtime or existing[1] != size:
                content_hash = compute_content_hash(path)
                if existing[2] != content_hash:
                    upserted += 1
                self.upsert_from_file(
                    path,
                    content_hash=content_hash,
                    mtime=mtime,
                    size=size,
                )

            if on_progress is not None and (
                (index + 1) % commit_batch_size == 0 or index + 1 == total
            ):
                on_progress(index + 1, total)

            if commit_batch_size > 0 and (index + 1) % commit_batch_size == 0:
                self._session.commit()

        removed = self._remove_missing(resolved_keys) if remove_missing else 0
        return ReconcileResult(upserted=upserted, removed=removed)

    def _remove_missing(self, active_path_keys: list[str]) -> int:
        active_strs = set(active_path_keys)
        db_paths = list(self._session.scalars(select(Image.path)))
        to_delete = [path for path in db_paths if path not in active_strs]
        if not to_delete:
            return 0

        removed = 0
        for offset in range(0, len(to_delete), _PATH_BATCH_SIZE):
            chunk = to_delete[offset : offset + _PATH_BATCH_SIZE]
            result = self._session.execute(delete(Image).where(Image.path.in_(chunk)))
            removed += int(result.rowcount or 0)
        return removed

    def _invalidate_indexed_data(self, image_id: str) -> None:
        self._session.execute(delete(Detection).where(Detection.image_id == image_id))
        self._session.execute(delete(Face).where(Face.image_id == image_id))
        self._session.execute(delete(ImageYolo).where(ImageYolo.image_id == image_id))
        self._session.execute(delete(ImageFaces).where(ImageFaces.image_id == image_id))
        self._session.execute(delete(ImageClip).where(ImageClip.image_id == image_id))
        if self._vector_store is not None and self._vector_store.available:
            self._vector_store.delete_for_image(image_id)
