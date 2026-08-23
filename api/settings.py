from fastapi import APIRouter, Request

from api.schemas import ScanConfigResponse, ScanConfigUpdate
from db.scan_config import ScanConfig

router = APIRouter(prefix="/settings", tags=["settings"])


def _to_response(config: ScanConfig) -> ScanConfigResponse:
    return ScanConfigResponse(
        include_directories=config.include_directories,
        ignore_globs=config.ignore_globs,
        background_indexer_enabled=config.background_indexer_enabled,
        schedule_interval_days=config.schedule_interval_days,
        background_modules=config.background_modules,
        last_background_run_at=config.last_background_run_at,
    )


@router.get("/scan", response_model=ScanConfigResponse)
def get_scan_settings(request: Request) -> ScanConfigResponse:
    with request.app.state.db:
        config = request.app.state.db.get_scan_config()
    return _to_response(config)


@router.put("/scan", response_model=ScanConfigResponse)
def put_scan_settings(request: Request, payload: ScanConfigUpdate) -> ScanConfigResponse:
    with request.app.state.db:
        current = request.app.state.db.get_scan_config()
        config = ScanConfig(
            include_directories=payload.include_directories
            if payload.include_directories is not None
            else current.include_directories,
            ignore_globs=payload.ignore_globs
            if payload.ignore_globs is not None
            else current.ignore_globs,
            background_indexer_enabled=(
                payload.background_indexer_enabled
                if payload.background_indexer_enabled is not None
                else current.background_indexer_enabled
            ),
            schedule_interval_days=(
                payload.schedule_interval_days
                if payload.schedule_interval_days is not None
                else current.schedule_interval_days
            ),
            background_modules=(
                payload.background_modules
                if payload.background_modules is not None
                else current.background_modules
            ),
            last_background_run_at=current.last_background_run_at,
        )
        request.app.state.db.scan_config.save(config)
    return _to_response(config)
