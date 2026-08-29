from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from db.database import Database
from db.scan_config import ScanConfig
from db.types import MODULE_CLIP, MODULE_FACES, MODULE_YOLO, ImageRecord, ModuleStatus
from indexing.clip import index_clip_image
from indexing.faces import index_faces_image, store_faces
from indexing.gap import clip_gap_paths, faces_gap_paths, module_gap_paths, yolo_gap_paths
from indexing.gpu_scheduler import GpuScheduler, GpuTaskCancelled, get_gpu_scheduler
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
_CLIP_BATCH_SIZE = max(1, int(os.environ.get("CLIP_BATCH_SIZE", "32")))
_YOLO_BATCH_SIZE = max(1, int(os.environ.get("YOLO_BATCH_SIZE", "16")))
_FACES_BATCH_SIZE = max(1, int(os.environ.get("FACES_BATCH_SIZE", "16")))
# CPU decode workers used inside ArcFace.analyze_batch. ONNX inference stays serial.


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


@dataclass(frozen=True)
class CatalogResult:
    total: int
    upserted: int
    removed: int


def collect_scope_paths(config: ScanConfig) -> list[Path]:
    return collect_scoped_files(config.include_paths(), config.ignore_globs)


def reconcile_catalog(
    db: Database,
    config: ScanConfig,
    *,
    run_id: str | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> CatalogResult:
    paths = collect_scope_paths(config)

    def on_progress(done: int, total: int) -> None:
        if should_stop is not None and should_stop():
            raise ScanStopped("catalog reconcile aborted by shutdown")
        if run_id is not None:
            db.index_runs.update_progress(run_id, progress_done=done)

    if run_id is not None:
        with db:
            db.index_runs.set_phase(
                run_id,
                "reconcile",
                progress_done=0,
                progress_total=len(paths),
            )

    with db:
        result = db.reconcile_paths(
            set(paths),
            remove_missing=True,
            on_progress=on_progress,
            commit_batch_size=_RECONCILE_COMMIT_BATCH,
        )
    return CatalogResult(
        total=len(paths),
        upserted=result.upserted,
        removed=result.removed,
    )


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


def _chunks(paths: list[Path], size: int):
    for offset in range(0, len(paths), size):
        yield offset, paths[offset : offset + size]


def _check_stop(should_stop: Callable[[], bool] | None) -> None:
    if should_stop is not None and should_stop():
        raise ScanStopped("scan aborted by shutdown")


def _run_clip_batches(
    db: Database,
    vector_store: VectorStore,
    model: EmbeddingModel,
    paths: list[Path],
    *,
    run_id: str | None,
    should_stop: Callable[[], bool] | None,
    scheduler: GpuScheduler,
) -> tuple[int, int]:
    indexed = 0
    failed = 0
    total = len(paths)
    for offset, batch in _chunks(paths, _CLIP_BATCH_SIZE):
        _check_stop(should_stop)
        try:
            with scheduler.acquire(
                "index:clip",
                priority=GpuScheduler.INDEXING,
                should_stop=should_stop,
            ):
                embeddings = model.encode_images(batch)
            if len(embeddings) != len(batch):
                raise RuntimeError("CLIP batch returned an unexpected number of embeddings")
            with db:
                records = db.images.map_records_by_path(batch)
                points = []
                for path, embedding in zip(batch, embeddings, strict=True):
                    record = records.get(str(path))
                    if record is None:
                        raise ValueError(f"image not registered: {path}")
                    points.append((record.id, embedding))
                vector_store.upsert_contexts(points, model_version=model.model_name)
                for record in records.values():
                    db.image_clip.mark_done(record.id, model_version=model.model_name)
            indexed += len(batch)
        except GpuTaskCancelled as exc:
            raise ScanStopped(str(exc)) from exc
        except Exception:
            logger.exception("failed to index CLIP batch at offset %d", offset)
            # Isolate corrupt files and preserve useful work from the batch.
            for path in batch:
                _check_stop(should_stop)
                try:
                    with scheduler.acquire(
                        "index:clip",
                        priority=GpuScheduler.INDEXING,
                        should_stop=should_stop,
                    ):
                        embedding = model.encode_image(path)
                    with db:
                        record = db.images.get_by_path(path)
                        if record is None:
                            raise ValueError(f"image not registered: {path}")
                        vector_store.upsert_context(
                            record.id,
                            embedding,
                            model_version=model.model_name,
                        )
                        db.image_clip.mark_done(record.id, model_version=model.model_name)
                    indexed += 1
                except GpuTaskCancelled as cancelled:
                    raise ScanStopped(str(cancelled)) from cancelled
                except Exception as item_exc:
                    logger.exception("failed to index clip for %s", path)
                    with db:
                        _mark_index_failed(db, path, MODULE_CLIP, str(item_exc))
                    failed += 1
        if run_id is not None:
            with db:
                db.index_runs.update_progress(
                    run_id,
                    progress_done=min(total, offset + len(batch)),
                )
    return indexed, failed


def _run_yolo_batches(
    db: Database,
    model: ObjectsRetriever,
    paths: list[Path],
    *,
    run_id: str | None,
    should_stop: Callable[[], bool] | None,
    scheduler: GpuScheduler,
) -> tuple[int, int]:
    indexed = 0
    failed = 0
    total = len(paths)
    for offset, batch in _chunks(paths, _YOLO_BATCH_SIZE):
        _check_stop(should_stop)
        try:
            with scheduler.acquire(
                "index:yolo",
                priority=GpuScheduler.INDEXING,
                should_stop=should_stop,
            ):
                detections_by_image = model.detect_batch(batch)
            if len(detections_by_image) != len(batch):
                raise RuntimeError("YOLO batch returned an unexpected number of results")
            # Inference deliberately happens before this short write transaction.
            with db:
                records = db.images.map_records_by_path(batch)
                for path, detections in zip(batch, detections_by_image, strict=True):
                    record = records.get(str(path))
                    if record is None:
                        raise ValueError(f"image not registered: {path}")
                    db.detections.replace_for_image(record.id, detections)
                    db.image_yolo.mark_done(record.id, model_version=model.model_name)
            indexed += len(batch)
        except GpuTaskCancelled as exc:
            raise ScanStopped(str(exc)) from exc
        except Exception:
            logger.exception("failed to index YOLO batch at offset %d", offset)
            for path in batch:
                _check_stop(should_stop)
                try:
                    with scheduler.acquire(
                        "index:yolo",
                        priority=GpuScheduler.INDEXING,
                        should_stop=should_stop,
                    ):
                        detections = model.detect(path)
                    with db:
                        record = db.images.get_by_path(path)
                        if record is None:
                            raise ValueError(f"image not registered: {path}")
                        db.detections.replace_for_image(record.id, detections)
                        db.image_yolo.mark_done(record.id, model_version=model.model_name)
                    indexed += 1
                except GpuTaskCancelled as cancelled:
                    raise ScanStopped(str(cancelled)) from cancelled
                except Exception as item_exc:
                    logger.exception("failed to index yolo for %s", path)
                    with db:
                        _mark_index_failed(db, path, MODULE_YOLO, str(item_exc))
                    failed += 1
        if run_id is not None:
            with db:
                db.index_runs.update_progress(
                    run_id,
                    progress_done=min(total, offset + len(batch)),
                )
    return indexed, failed


def _run_faces_batches(
    db: Database,
    vector_store: VectorStore,
    recognizer: FaceRecognizer,
    paths: list[Path],
    *,
    run_id: str | None,
    should_stop: Callable[[], bool] | None,
    scheduler: GpuScheduler,
) -> tuple[int, int]:
    indexed = 0
    failed = 0
    total = len(paths)
    model_version = recognizer.model_name
    for offset, batch in _chunks(paths, _FACES_BATCH_SIZE):
        _check_stop(should_stop)
        try:
            with scheduler.acquire(
                "index:faces",
                priority=GpuScheduler.INDEXING,
                should_stop=should_stop,
            ):
                faces_by_image = recognizer.analyze_batch(
                    batch,
                    should_stop=should_stop,
                )
            if len(faces_by_image) != len(batch):
                raise RuntimeError("face batch returned an unexpected number of results")
            with db:
                for path, faces in zip(batch, faces_by_image, strict=True):
                    store_faces(
                        db,
                        vector_store,
                        path,
                        faces,
                        model_version=model_version,
                    )
            indexed += len(batch)
        except InterruptedError as exc:
            raise ScanStopped(str(exc)) from exc
        except GpuTaskCancelled as exc:
            raise ScanStopped(str(exc)) from exc
        except Exception:
            logger.exception("failed to index faces batch at offset %d", offset)
            for path in batch:
                _check_stop(should_stop)
                try:
                    with scheduler.acquire(
                        "index:faces",
                        priority=GpuScheduler.INDEXING,
                        should_stop=should_stop,
                    ):
                        faces = recognizer.analyze(path)
                    with db:
                        store_faces(
                            db,
                            vector_store,
                            path,
                            faces,
                            model_version=model_version,
                        )
                    indexed += 1
                except InterruptedError as cancelled:
                    raise ScanStopped(str(cancelled)) from cancelled
                except GpuTaskCancelled as cancelled:
                    raise ScanStopped(str(cancelled)) from cancelled
                except Exception as item_exc:
                    logger.exception("failed to index faces for %s", path)
                    with db:
                        _mark_index_failed(db, path, MODULE_FACES, str(item_exc))
                    failed += 1
        if run_id is not None:
            with db:
                db.index_runs.update_progress(
                    run_id,
                    progress_done=min(total, offset + len(batch)),
                )
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
    scheduler: GpuScheduler | None = None,
) -> list[ScanResult]:
    if len(modules) != 1 and mode == "full":
        raise ValueError("full run supports exactly one module per invocation")

    del remove_missing  # Catalog reconciliation is an explicit, separate job.
    scheduler = scheduler or get_gpu_scheduler()
    with db:
        paths = [record.path for record in db.images.list_all()]
    results: list[ScanResult] = []

    logger.info(
        "starting scan mode=%s modules=%s catalog_paths=%d",
        mode,
        modules,
        len(paths),
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

        if module == MODULE_CLIP:
            indexed, failed = _run_clip_batches(
                db,
                vector_store,
                models.clip,
                gap_paths,
                run_id=run_id,
                should_stop=should_stop,
                scheduler=scheduler,
            )
        elif module == MODULE_YOLO:
            indexed, failed = _run_yolo_batches(
                db,
                models.yolo,
                gap_paths,
                run_id=run_id,
                should_stop=should_stop,
                scheduler=scheduler,
            )
        elif module == MODULE_FACES:
            indexed, failed = _run_faces_batches(
                db,
                vector_store,
                models.faces,
                gap_paths,
                run_id=run_id,
                should_stop=should_stop,
                scheduler=scheduler,
            )
        else:
            for index, path in enumerate(gap_paths):
                if should_stop is not None and should_stop():
                    logger.info("scan stop requested; aborting module=%s at %d/%d", module, index, total)
                    raise ScanStopped("scan aborted by shutdown")
                try:
                    try:
                        with scheduler.acquire(
                            f"index:{module}",
                            priority=GpuScheduler.INDEXING,
                            should_stop=should_stop,
                        ):
                            with db:
                                _index_single(db, vector_store, models, path, module)
                    except GpuTaskCancelled as exc:
                        raise ScanStopped(str(exc)) from exc
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
