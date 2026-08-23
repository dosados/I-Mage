from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from api.schemas import (
    FacesReadyResponse,
    IndexRunResponse,
    IndexStatusResponse,
    ModuleRunStatus,
)
from db.types import MODULE_CLIP, MODULE_FACES, MODULE_YOLO
from indexing.executor import IndexRunConflictError, IndexExecutor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/index", tags=["index"])

VALID_MODULES = {MODULE_YOLO, MODULE_CLIP, MODULE_FACES}
_ALL_MODULES = (MODULE_YOLO, MODULE_CLIP, MODULE_FACES)


def _get_executor(request: Request) -> IndexExecutor:
    executor = getattr(request.app.state, "index_executor", None)
    if executor is None:
        raise HTTPException(status_code=503, detail="index executor not ready")
    return executor


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _background_status(request: Request) -> dict:
    with request.app.state.db:
        config = request.app.state.db.get_scan_config()
    next_run_at = None
    if config.background_indexer_enabled:
        last_run = _parse_iso(config.last_background_run_at)
        if last_run is None:
            next_run_at = datetime.now(timezone.utc).isoformat()
        else:
            next_run_at = (last_run + timedelta(days=config.schedule_interval_days)).isoformat()
    return {
        "enabled": config.background_indexer_enabled,
        "schedule_interval_days": config.schedule_interval_days,
        "background_modules": config.background_modules,
        "last_background_run_at": config.last_background_run_at,
        "next_run_at": next_run_at,
    }


def _run_response(record) -> IndexRunResponse | None:
    if record is None:
        return None
    total = record.progress_total or 0
    if record.status == "done":
        percent = 100
    elif record.phase == "pending" or total <= 0:
        percent = 0
    else:
        percent = min(100, round(record.progress_done / total * 100))
    return IndexRunResponse(
        id=record.id,
        module=record.module,
        mode=record.mode,
        status=record.status,
        phase=record.phase,
        progress_done=record.progress_done,
        progress_total=record.progress_total,
        percent=percent,
        started_at=record.started_at,
        finished_at=record.finished_at,
        last_error=record.last_error,
    )


@router.post("/run/full/{module}", status_code=202)
async def run_full_module(module: str, request: Request) -> JSONResponse:
    if module not in VALID_MODULES:
        raise HTTPException(status_code=400, detail=f"module must be one of {sorted(VALID_MODULES)}")

    executor = _get_executor(request)
    try:
        run_id = await executor.start_full_run(module)
    except IndexRunConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=503 if "Qdrant" in str(exc) else 400, detail=str(exc)) from exc

    return JSONResponse(
        status_code=202,
        content={"run_id": run_id, "module": module, "mode": "full"},
    )


@router.post("/run/background", status_code=202)
async def run_background_gap(request: Request) -> JSONResponse:
    executor = _get_executor(request)
    try:
        await executor.start_background_gap()
    except IndexRunConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return JSONResponse(status_code=202, content={"status": "started", "mode": "gap"})


@router.get("/status", response_model=IndexStatusResponse)
def index_status(
    request: Request,
    module: str | None = None,
    stats: bool = Query(
        default=True,
        description="Include per-module catalog stats (SQL COUNTs; no filesystem walk)",
    ),
) -> IndexStatusResponse:
    if module is not None and module not in VALID_MODULES:
        raise HTTPException(status_code=400, detail=f"module must be one of {sorted(VALID_MODULES)}")

    with request.app.state.db:
        active = request.app.state.db.index_runs.get_active(module=module)
        latest = request.app.state.db.index_runs.get_latest(module=module)
        scope_total = request.app.state.db.images.count_all()

        # Always return per-module runs so the UI can poll once instead of
        # firing 3 parallel /status?module=… requests (that pattern held SQLite
        # locks for ~10s on a 30k catalog).
        module_runs = {
            mod: ModuleRunStatus(
                active_run=_run_response(
                    request.app.state.db.index_runs.get_active(module=mod)
                ),
                latest_run=_run_response(
                    request.app.state.db.index_runs.get_latest(module=mod)
                ),
            )
            for mod in _ALL_MODULES
        }

        modules_stats: dict[str, dict[str, int]] = {}
        if stats:
            # Fast catalog COUNTs — previously this walked the whole dataset on
            # disk and ran path-IN queries for every module on every poll.
            modules_stats = request.app.state.db.images.catalog_module_stats()

    return IndexStatusResponse(
        active_run=_run_response(active),
        latest_run=_run_response(latest),
        modules=modules_stats,
        module_runs=module_runs,
        background=_background_status(request),
        scope_total=scope_total,
    )


@router.get("/faces-ready", response_model=FacesReadyResponse)
def faces_ready(request: Request) -> FacesReadyResponse:
    with request.app.state.db:
        total = request.app.state.db.images.count_all()
        done = request.app.state.db.images.count_module_done(MODULE_FACES)
        # Relaxed from all-or-nothing: as soon as some faces are indexed the
        # people panel is usable. A single failed image no longer blocks it.
        ready = total > 0 and done > 0
    return FacesReadyResponse(ready=ready, done=done, total=total)
