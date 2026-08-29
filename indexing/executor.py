from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING

from db.database import Database
from db.types import MODULE_CLIP, MODULE_FACES, MODULE_YOLO
from indexing.cluster import run_face_clustering
from indexing.gpu_scheduler import GpuScheduler, get_gpu_scheduler
from indexing.runner import (
    IndexModels,
    ScanResult,
    ScanStopped,
    reconcile_catalog,
    run_scan,
)

if TYPE_CHECKING:
    from vectors.store import VectorStore

logger = logging.getLogger(__name__)

VALID_MODULES = frozenset({MODULE_YOLO, MODULE_CLIP, MODULE_FACES})
CATALOG_MODULE = "catalog"


class IndexRunConflictError(RuntimeError):
    pass


class IndexExecutor:
    """Coordinates catalog jobs, index runs and the shared GPU scheduler."""

    def __init__(
        self,
        db: Database,
        vector_store: VectorStore,
        models: IndexModels,
        scheduler: GpuScheduler | None = None,
    ) -> None:
        self._db = db
        self._vector_store = vector_store
        self._models = models
        self._scheduler = scheduler or get_gpu_scheduler()
        self._module_locks = {module: threading.Lock() for module in VALID_MODULES}
        self._catalog_lock = threading.Lock()
        self._tasks: dict[str, asyncio.Task] = {}
        self._stop_event = threading.Event()

    def has_manual_run_in_progress(self) -> bool:
        return self._catalog_lock.locked() or any(
            lock.locked() for lock in self._module_locks.values()
        )

    def gpu_status(self) -> dict[str, object]:
        return self._scheduler.snapshot()

    async def shutdown(self, *, timeout: float = 30.0) -> None:
        """Signal in-flight scans to stop and wait for their tasks to finish.

        Must be called before closing the DB / vector store so running scans
        don't operate on closed resources.
        """
        self._stop_event.set()
        tasks = [task for task in self._tasks.values() if not task.done()]
        if not tasks:
            return
        logger.info("waiting for %d index task(s) to stop", len(tasks))
        _done, pending = await asyncio.wait(tasks, timeout=timeout)
        for task in pending:
            logger.warning("index task did not stop within %.0fs; cancelling", timeout)
            task.cancel()

    async def start_full_run(self, module: str) -> str:
        if module not in VALID_MODULES:
            raise ValueError(f"unknown module: {module}")
        if self._catalog_lock.locked():
            raise IndexRunConflictError("catalog reconcile already in progress")

        lock = self._module_locks[module]
        if not lock.acquire(blocking=False):
            raise IndexRunConflictError(f"{module} index run already in progress")

        try:
            run_id = await asyncio.to_thread(self._create_run, module)
        except Exception:
            lock.release()
            raise

        task = asyncio.create_task(self._run_full(module, run_id, lock))
        self._tasks[run_id] = task
        task.add_done_callback(lambda _task: self._tasks.pop(run_id, None))
        return run_id

    async def start_cluster_run(self, *, regroup: bool) -> str:
        """Start a clustering-only run (manual "Найти людей" / "Перегруппировать").

        Shares the faces module lock so it can't overlap a faces index run, and
        is tracked as an index_runs row (phase=clustering) so the UI reads its
        progress from /index/status like any other phase.
        """
        if self._catalog_lock.locked():
            raise IndexRunConflictError("catalog reconcile already in progress")
        lock = self._module_locks[MODULE_FACES]
        if not lock.acquire(blocking=False):
            raise IndexRunConflictError("faces index run already in progress")

        try:
            run_id = await asyncio.to_thread(self._create_cluster_run)
        except Exception:
            lock.release()
            raise

        task = asyncio.create_task(self._run_cluster(run_id, lock, regroup))
        self._tasks[run_id] = task
        task.add_done_callback(lambda _task: self._tasks.pop(run_id, None))
        return run_id

    async def start_catalog_reconcile(self) -> str:
        if not self._catalog_lock.acquire(blocking=False):
            raise IndexRunConflictError("catalog reconcile already in progress")
        if any(lock.locked() for lock in self._module_locks.values()):
            self._catalog_lock.release()
            raise IndexRunConflictError("index run already in progress")
        try:
            run_id = await asyncio.to_thread(self._create_catalog_run)
        except Exception:
            self._catalog_lock.release()
            raise
        task = asyncio.create_task(self._run_catalog_reconcile(run_id))
        self._tasks[run_id] = task
        task.add_done_callback(lambda _task: self._tasks.pop(run_id, None))
        return run_id

    async def start_background_gap(self) -> None:
        existing = self._tasks.get("background-gap")
        if existing is not None and not existing.done():
            raise IndexRunConflictError("background index run already in progress")
        if self._catalog_lock.locked() or any(
            lock.locked() for lock in self._module_locks.values()
        ):
            raise IndexRunConflictError("manual index run already in progress")

        with self._db:
            if self._db.index_runs.get_active() is not None:
                raise IndexRunConflictError("index run already in progress")

        task = asyncio.create_task(self._run_background_gap())
        self._tasks["background-gap"] = task
        task.add_done_callback(lambda _task: self._tasks.pop("background-gap", None))

    def _ensure_vector_store(self) -> None:
        if self._vector_store.available:
            return
        self._vector_store._connect()
        if not self._vector_store.available:
            reason = self._vector_store.last_error or "Qdrant недоступен — проверьте /health"
            raise ValueError(reason)

    def _create_run(self, module: str) -> str:
        if module in (MODULE_CLIP, MODULE_FACES):
            self._ensure_vector_store()

        with self._db:
            active = self._db.index_runs.get_active(module=module)
            if active is not None:
                raise IndexRunConflictError(f"{module} index run already in progress")

            run = self._db.index_runs.create(
                module=module,
                mode="full",
                progress_total=0,
            )
            return run.id

    def _create_cluster_run(self) -> str:
        self._ensure_vector_store()
        with self._db:
            active = self._db.index_runs.get_active(module=MODULE_FACES)
            if active is not None:
                raise IndexRunConflictError("faces index run already in progress")
            run = self._db.index_runs.create(
                module=MODULE_FACES,
                mode="cluster",
                progress_total=100,
                phase="clustering",
            )
            return run.id

    def _create_catalog_run(self) -> str:
        with self._db:
            if self._db.index_runs.get_active() is not None:
                raise IndexRunConflictError("index run already in progress")
            run = self._db.index_runs.create(
                module=CATALOG_MODULE,
                mode="reconcile",
                progress_total=0,
            )
            return run.id

    @staticmethod
    def _finish_run(db: Database, run_id: str, results: list[ScanResult]) -> None:
        result = results[0] if len(results) == 1 else None
        if result is None or result.total == 0:
            db.index_runs.mark_done(run_id)
            return

        if result.indexed == 0:
            db.index_runs.mark_failed(
                run_id,
                f"Ни одно фото не проиндексировано ({result.failed} из {result.total} с ошибкой)",
            )
            return

        if result.failed > 0:
            db.index_runs.mark_done(
                run_id,
                summary=f"Завершено с ошибками: {result.indexed} ок, {result.failed} не удалось",
            )
            return

        db.index_runs.mark_done(run_id)

    async def _run_full(self, module: str, run_id: str, lock: threading.Lock) -> None:
        try:
            await asyncio.to_thread(
                self._execute_scan,
                module=module,
                run_id=run_id,
                mode="full",
                remove_missing=True,
            )
        finally:
            lock.release()

    async def _run_background_gap(self) -> None:
        await asyncio.to_thread(
            self._execute_scan,
            module=None,
            run_id=None,
            mode="gap",
            remove_missing=False,
            background=True,
        )

    async def _run_cluster(self, run_id: str, lock: threading.Lock, regroup: bool) -> None:
        try:
            await asyncio.to_thread(self._execute_cluster, run_id=run_id, regroup=regroup)
        finally:
            lock.release()

    async def _run_catalog_reconcile(self, run_id: str) -> None:
        try:
            await asyncio.to_thread(self._execute_catalog_reconcile, run_id)
        finally:
            self._catalog_lock.release()

    def _execute_cluster(self, *, run_id: str, regroup: bool) -> None:
        try:
            run_face_clustering(
                self._db,
                self._vector_store,
                run_id=run_id,
                regroup=regroup,
                should_stop=self._stop_event.is_set,
            )
            with self._db:
                self._db.index_runs.mark_done(run_id)
        except ScanStopped:
            logger.info("cluster run stopped (run_id=%s)", run_id)
            with self._db:
                self._db.index_runs.mark_failed(
                    run_id, "Остановлено при завершении сервера — запустите заново"
                )
        except Exception as exc:
            logger.exception("cluster run failed (run_id=%s)", run_id)
            with self._db:
                self._db.index_runs.mark_failed(run_id, str(exc))
            raise

    def _execute_catalog_reconcile(self, run_id: str) -> None:
        try:
            with self._db:
                config = self._db.get_scan_config()
            result = reconcile_catalog(
                self._db,
                config,
                run_id=run_id,
                should_stop=self._stop_event.is_set,
            )
            with self._db:
                self._db.index_runs.mark_done(
                    run_id,
                    summary=(
                        f"Каталог: {result.total}; обновлено: {result.upserted}; "
                        f"удалено: {result.removed}"
                    ),
                )
        except ScanStopped:
            with self._db:
                self._db.index_runs.mark_failed(
                    run_id, "Остановлено при завершении сервера — запустите заново"
                )
        except Exception as exc:
            logger.exception("catalog reconcile failed (run_id=%s)", run_id)
            with self._db:
                self._db.index_runs.mark_failed(run_id, str(exc))
            raise

    def _execute_scan(
        self,
        *,
        module: str | None,
        run_id: str | None,
        mode: str,
        remove_missing: bool,
        background: bool = False,
    ) -> None:
        try:
            with self._db:
                config = self._db.get_scan_config()

            if background:
                modules = [
                    item
                    for item in config.background_modules
                    if item in VALID_MODULES
                ]
            else:
                modules = [module] if module is not None else []

            if not modules:
                if run_id is not None:
                    with self._db:
                        self._db.index_runs.mark_done(run_id)
                return

            results = run_scan(
                self._db,
                self._vector_store,
                self._models,
                config,
                modules=modules,
                mode=mode,
                remove_missing=remove_missing,
                run_id=run_id,
                should_stop=self._stop_event.is_set,
                scheduler=self._scheduler,
            )

            # Face clustering remains part of the tracked faces run, but catalog
            # reconciliation is now an explicit, independent operation.
            if run_id is not None and not background and MODULE_FACES in modules:
                run_face_clustering(
                    self._db,
                    self._vector_store,
                    run_id=run_id,
                    regroup=False,
                    should_stop=self._stop_event.is_set,
                )

            with self._db:
                if run_id is not None:
                    self._finish_run(self._db, run_id, results)
                if background:
                    from db.models import utc_now_iso

                    self._db.scan_config.update_last_background_run(utc_now_iso())
        except ScanStopped:
            logger.info("index run stopped (run_id=%s, module=%s)", run_id, module)
            if run_id is not None:
                with self._db:
                    self._db.index_runs.mark_failed(
                        run_id, "Остановлено при завершении сервера — запустите прогон заново"
                    )
            return
        except Exception as exc:
            logger.exception("index run failed (run_id=%s, module=%s)", run_id, module)
            if run_id is not None:
                with self._db:
                    self._db.index_runs.mark_failed(run_id, str(exc))
            raise
