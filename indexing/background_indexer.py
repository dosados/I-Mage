from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from db.database import Database
from db.scan_config import ScanConfig
from indexing.executor import IndexExecutor, IndexRunConflictError
from indexing.runner import IndexModels

logger = logging.getLogger(__name__)

CHECK_INTERVAL_SECONDS = 3600


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _next_background_run_at(config: ScanConfig) -> datetime | None:
    if not config.background_indexer_enabled:
        return None
    last_run = _parse_iso(config.last_background_run_at)
    if last_run is None:
        return datetime.now(timezone.utc)
    return last_run + timedelta(days=config.schedule_interval_days)


async def background_indexer_loop(
    *,
    db: Database,
    vector_store,
    models: IndexModels,
    stop_event: asyncio.Event,
    executor: IndexExecutor | None = None,
) -> None:
    while not stop_event.is_set():
        try:
            with db:
                config = db.get_scan_config()

            if config.background_indexer_enabled:
                next_run = _next_background_run_at(config)
                now = datetime.now(timezone.utc)
                if next_run is not None and now >= next_run:
                    if executor is None:
                        logger.warning("background indexer has no executor; skipping tick")
                    else:
                        # Route through the executor so the scan is tracked in
                        # _tasks (awaited + cooperatively stopped on shutdown)
                        # and shares the same locks as manual runs.
                        try:
                            await executor.start_background_gap()
                        except IndexRunConflictError:
                            logger.info("background indexer skipped: run already in progress")
        except Exception:
            logger.exception("background indexer tick failed")

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=CHECK_INTERVAL_SECONDS)
        except asyncio.TimeoutError:
            continue
