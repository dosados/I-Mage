from __future__ import annotations

import concurrent.futures as cf
import logging
import os
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from db.database import Database
from db.scan_config import ScanConfig
from db.types import MODULE_CLIP, MODULE_FACES, MODULE_YOLO, ImageRecord, ModuleStatus
from indexing.clip import index_clip_image
from indexing.faces import index_faces_image, store_faces
from indexing.gap import clip_gap_paths, faces_gap_paths, module_gap_paths, yolo_gap_paths
from indexing.yolo import index_yolo_image
from io_utils.scan import collect_scoped_files
from ml.embeddings.base import EmbeddingModel
from ml.faces.base import FaceRecognizer
from ml.objects.base import ObjectsRetriever
from vectors.store import VectorStore

logger = logging.getLogger(__name__)

# Reconcile commits: larger batches cut SQLite fsync overhead on 30k catalogs.
# Progress is still visible because the UI polls ~1s and we report every batch.
_RECONCILE_COMMIT_BATCH = 500
_INDEX_PROGRESS_BATCH = 10
# Face analysis (detect+recognize) is CPU-prep + short GPU bursts, so a single
# image barely uses the GPU. Running analysis in a small thread pool overlaps the
# CPU preprocessing of several images and keeps the GPU fed, while DB/Qdrant
# writes stay on one thread. Tune via FACES_ANALYZE_WORKERS (1 = old serial path).
_FACES_WORKERS = max(1, int(os.environ.get("FACES_ANALYZE_WORKERS", "4")))


class ScanStopped(RuntimeError):
    """Raised to cooperatively abort a scan (e.g. on server shutdown)."""


@dataclass(frozen=True)
class IndexModels:
    clip: EmbeddingModel
    yolo: ObjectsRetriever
    faces: FaceRecognizer


@dataclass(frozen=True)
class ScanResult:
    module: str
    mode: str
    total: int
    indexed: int
    failed: int
    run_id: str | None = None


def collect_scope_paths(config: ScanConfig) -> list[Path]:
    return collect_scoped_files(config.include_paths(), config.ignore_globs)


def _module_done(record: ImageRecord | None, module: str) -> bool:
    if record is None:
        return False
    module_index = record.modules.get(module)
    return module_index is not None and module_index.status == ModuleStatus.DONE


def faces_ready_in_scope(
    db: Database,
    paths: list[Path],
    *,
    records_by_path: dict[str, ImageRecord] | None = None,
) -> bool:
    from db.hash import resolved_path_key

    if not paths:
        return False
    if records_by_path is None:
        with db:
            records_by_path = db.images.map_records_by_path(paths)
    for path in paths:
        if not _module_done(records_by_path.get(resolved_path_key(path)), MODULE_FACES):
            return False
    return True


def module_stats_in_scope(
    db: Database,
    paths: list[Path],
    module: str,
    *,
    records_by_path: dict[str, ImageRecord] | None = None,
) -> tuple[int, int]:
    from db.hash import resolved_path_key

    if not paths:
        return 0, 0
    if records_by_path is not None:
        done = sum(
            1
            for path in paths
            if _module_done(records_by_path.get(resolved_path_key(path)), module)
        )
        return done, len(paths)
    with db:
        done = db.images.count_module_done_in_paths(paths, module)
    return done, len(paths)


def _mark_index_failed(db: Database, path: Path, module: str, error: str) -> None:
    record = db.images.get_by_path(path)
    if record is None:
        return

    if module == MODULE_YOLO:
        db.image_yolo.mark_failed(record.id, error)
    elif module == MODULE_CLIP:
        db.image_clip.mark_failed(record.id, error)
    elif module == MODULE_FACES:
        db.image_faces.mark_failed(record.id, error)


def _index_single(
    db: Database,
    vector_store: VectorStore,
    models: IndexModels,
    path: Path,
    module: str,
) -> None:
    if module == MODULE_YOLO:
        index_yolo_image(db, path, models.yolo, model_version=models.yolo.model_name)
        return
    if module == MODULE_CLIP:
        if not vector_store.available:
            raise RuntimeError("qdrant is not available")
        index_clip_image(
            db,
            vector_store,
            path,
            models.clip,
            model_version=models.clip.model_name,
        )
        return
    if module == MODULE_FACES:
        if not vector_store.available:
            raise RuntimeError("qdrant is not available")
        index_faces_image(
            db,
            vector_store,
            path,
            models.faces,
            model_version=models.faces.model_name,
        )
        return
    raise ValueError(f"unknown module: {module}")


def _gap_paths_for_module(
    paths: list[Path],
    module: str,
    *,
    records_by_path: dict[str, ImageRecord],
) -> list[Path]:
    if module == MODULE_YOLO:
        return yolo_gap_paths(None, paths, records_by_path=records_by_path)
    if module == MODULE_CLIP:
        return clip_gap_paths(None, paths, records_by_path=records_by_path)
    if module == MODULE_FACES:
        return faces_gap_paths(None, paths, records_by_path=records_by_path)
    return module_gap_paths(None, paths, module, records_by_path=records_by_path)


def _index_faces_parallel(
    db: Database,
    vector_store: VectorStore,
    recognizer: FaceRecognizer,
    gap_paths: list[Path],
    *,
    model_version: str,
    run_id: str | None,
    should_stop: "Callable[[], bool] | None",
    workers: int,
    progress_batch: int,
) -> tuple[int, int]:
    """Analyze faces in a thread pool; persist results in submission order.

    Analysis (GPU/CPU) runs concurrently across ``workers`` threads while this
    (single) thread does all DB + Qdrant writes, so ordering and transactions
    stay correct and thread-safe.
    """
    indexed = 0
    failed = 0
    total = len(gap_paths)
    numbered = enumerate(gap_paths)

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        inflight: deque[tuple[int, Path, cf.Future]] = deque()

        def submit_next() -> bool:
            try:
                idx, path = next(numbered)
            except StopIteration:
                return False
            inflight.append((idx, path, pool.submit(recognizer.analyze, path)))
            return True

        # Keep the pool primed with a bit more work than it has threads.
        for _ in range(workers * 2):
            if not submit_next():
                break

        while inflight:
            if should_stop is not None and should_stop():
                for _, _, pending in inflight:
                    pending.cancel()
                raise ScanStopped("scan aborted by shutdown")

            index, path, future = inflight.popleft()
            submit_next()

            try:
                faces = future.result()
                with db:
                    store_faces(db, vector_store, path, faces, model_version=model_version)
                indexed += 1
            except Exception as exc:  # noqa: BLE001
                logger.exception("failed to index faces for %s", path)
                with db:
                    _mark_index_failed(db, path, MODULE_FACES, str(exc))
                failed += 1

            if run_id is not None and (
                (index + 1) % progress_batch == 0 or (index + 1) == total
            ):
                with db:
                    db.index_runs.update_progress(run_id, progress_done=index + 1)

    return indexed, failed


def run_scan(
    db: Database,
    vector_store: VectorStore,
    models: IndexModels,
    config: ScanConfig,
    *,
    modules: list[str],
    mode: str,
    remove_missing: bool = False,
    run_id: str | None = None,
    should_stop: "Callable[[], bool] | None" = None,
) -> list[ScanResult]:
    if len(modules) != 1 and mode == "full":
        raise ValueError("full run supports exactly one module per invocation")

    # Enter the reconcile phase BEFORE the filesystem walk so the UI shows stage 1
    # as active immediately instead of a blank "pending" while we scan the disk.
    if run_id is not None:
        with db:
            db.index_runs.set_phase(run_id, "reconcile", progress_done=0, progress_total=0)

    paths = collect_scope_paths(config)
    # collect_scope_paths already returns absolute resolved Paths.
    path_set = set(paths)
    results: list[ScanResult] = []

    def on_reconcile_progress(done: int, total: int) -> None:
        if should_stop is not None and should_stop():
            raise ScanStopped("scan aborted by shutdown")
        if run_id is None:
            return
        # Stay inside reconcile's transaction — do NOT commit here. A second
        # commit per batch doubled fsyncs and fought status-poll readers.
        if done == 0:
            db.index_runs.set_progress_total(run_id, total)
        db.index_runs.update_progress(run_id, progress_done=done)

    logger.info(
        "starting scan mode=%s modules=%s paths=%d remove_missing=%s",
        mode,
        modules,
        len(paths),
        remove_missing,
    )

    with db:
        db.reconcile_paths(
            path_set,
            remove_missing=remove_missing,
            on_progress=on_reconcile_progress,
            commit_batch_size=_RECONCILE_COMMIT_BATCH,
        )

    for module in modules:
        # Gap = scope paths whose module status isn't DONE. One flat query instead
        # of loading the whole catalog's ORM graph with joins.
        with db:
            done_paths = db.images.done_paths_for_module(module, paths)
        gap_paths = [path for path in paths if str(path) not in done_paths]

        total = len(gap_paths)
        indexed = 0
        failed = 0

        logger.info("module=%s gap_paths=%d", module, total)

        if run_id is not None:
            with db:
                db.index_runs.set_phase(
                    run_id, "indexing", progress_done=0, progress_total=total
                )

        if total == 0:
            results.append(
                ScanResult(
                    module=module,
                    mode=mode,
                    total=0,
                    indexed=0,
                    failed=0,
                    run_id=run_id,
                )
            )
            continue

        if module in (MODULE_CLIP, MODULE_FACES) and not vector_store.available:
            raise RuntimeError("qdrant is not available")

        if module == MODULE_FACES and _FACES_WORKERS > 1:
            indexed, failed = _index_faces_parallel(
                db,
                vector_store,
                models.faces,
                gap_paths,
                model_version=models.faces.model_name,
                run_id=run_id,
                should_stop=should_stop,
                workers=_FACES_WORKERS,
                progress_batch=_INDEX_PROGRESS_BATCH,
            )
        else:
            for index, path in enumerate(gap_paths):
                if should_stop is not None and should_stop():
                    logger.info("scan stop requested; aborting module=%s at %d/%d", module, index, total)
                    raise ScanStopped("scan aborted by shutdown")
                try:
                    with db:
                        _index_single(db, vector_store, models, path, module)
                    indexed += 1
                except Exception as exc:
                    logger.exception("failed to index %s for %s", module, path)
                    with db:
                        _mark_index_failed(db, path, module, str(exc))
                    failed += 1
                if run_id is not None and (
                    (index + 1) % _INDEX_PROGRESS_BATCH == 0 or (index + 1) == total
                ):
                    with db:
                        db.index_runs.update_progress(run_id, progress_done=index + 1)

        logger.info(
            "finished module=%s indexed=%d failed=%d total=%d",
            module,
            indexed,
            failed,
            total,
        )

        results.append(
            ScanResult(
                module=module,
                mode=mode,
                total=total,
                indexed=indexed,
                failed=failed,
                run_id=run_id,
            )
        )

    return results
