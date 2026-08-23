from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import AppSetting
from db.types import MODULE_CLIP, MODULE_FACES, MODULE_YOLO

SCAN_CONFIG_KEY = "scan_config"


@dataclass
class ScanConfig:
    include_directories: list[str] = field(default_factory=list)
    ignore_globs: list[str] = field(default_factory=list)
    background_indexer_enabled: bool = False
    schedule_interval_days: int = 7
    background_modules: list[str] = field(
        default_factory=lambda: [MODULE_YOLO, MODULE_CLIP, MODULE_FACES]
    )
    last_background_run_at: str | None = None

    def include_paths(self) -> list[Path]:
        return [Path(path).expanduser().resolve() for path in self.include_directories]


def default_scan_config(
    *,
    search_dir: Path,
    face_search_dir: Path,
) -> ScanConfig:
    directories: list[str] = []
    for directory in (search_dir, face_search_dir):
        resolved = str(directory.expanduser().resolve())
        if resolved not in directories and directory.is_dir():
            directories.append(resolved)
    if not directories:
        directories = [str(search_dir.expanduser().resolve())]
    return ScanConfig(include_directories=directories)


class ScanConfigService:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, *, defaults: ScanConfig) -> ScanConfig:
        row = self._session.get(AppSetting, SCAN_CONFIG_KEY)
        if row is None:
            return defaults
        try:
            payload = json.loads(row.value)
        except json.JSONDecodeError:
            return defaults
        return ScanConfig(
            include_directories=list(payload.get("include_directories", defaults.include_directories)),
            ignore_globs=list(payload.get("ignore_globs", defaults.ignore_globs)),
            background_indexer_enabled=bool(
                payload.get("background_indexer_enabled", defaults.background_indexer_enabled)
            ),
            schedule_interval_days=int(
                payload.get("schedule_interval_days", defaults.schedule_interval_days)
            ),
            background_modules=list(
                payload.get("background_modules", defaults.background_modules)
            ),
            last_background_run_at=payload.get("last_background_run_at"),
        )

    def save(self, config: ScanConfig) -> ScanConfig:
        payload = json.dumps(asdict(config), ensure_ascii=False)
        row = self._session.get(AppSetting, SCAN_CONFIG_KEY)
        if row is None:
            row = AppSetting(key=SCAN_CONFIG_KEY, value=payload)
            self._session.add(row)
        else:
            row.value = payload
        self._session.flush()
        return config

    def update_last_background_run(self, timestamp: str) -> ScanConfig:
        defaults = ScanConfig()
        config = self.get(defaults=defaults)
        config = ScanConfig(
            include_directories=config.include_directories,
            ignore_globs=config.ignore_globs,
            background_indexer_enabled=config.background_indexer_enabled,
            schedule_interval_days=config.schedule_interval_days,
            background_modules=config.background_modules,
            last_background_run_at=timestamp,
        )
        return self.save(config)
