from __future__ import annotations

import time
from pathlib import Path

from db.database import Database
from db.hash import compute_content_hash, read_file_stat
from db.scan_config import ScanConfig


def make_image_file(directory: Path, name: str, payload: bytes | None = None) -> Path:
    path = directory / name
    path.write_bytes(payload if payload is not None else name.encode("utf-8"))
    return path.resolve()


def register_file(db: Database, path: Path):
    content_hash = compute_content_hash(path)
    mtime, size = read_file_stat(path)
    return db.images.upsert_from_file(
        path, content_hash=content_hash, mtime=mtime, size=size
    )


def seed_catalog(
    db: Database,
    directory: Path,
    count: int,
    *,
    prefix: str = "img",
    mark_clip_done: int = 0,
    mark_yolo_done: int = 0,
    mark_faces_done: int = 0,
) -> list[Path]:
    """Create ``count`` tiny files and register them in the DB."""
    paths: list[Path] = []
    with db:
        for i in range(count):
            path = make_image_file(directory, f"{prefix}_{i:05d}.jpg", f"payload-{i}".encode())
            record = register_file(db, path)
            paths.append(path)
            if i < mark_yolo_done:
                db.image_yolo.mark_done(record.id, model_version="test")
            if i < mark_clip_done:
                db.image_clip.mark_done(record.id, model_version="test")
            if i < mark_faces_done:
                db.image_faces.mark_done(record.id, model_version="test")
    return paths


def configure_scan(db: Database, directories: list[Path]) -> ScanConfig:
    config = ScanConfig(include_directories=[str(d.resolve()) for d in directories])
    with db:
        db.set_default_scan_config(config)
        db.scan_config.save(config)
    return config


class LatencyRecorder:
    """Collect wall-time samples and print a short summary."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.samples_ms: list[float] = []

    def measure(self, fn):
        t0 = time.perf_counter()
        result = fn()
        self.samples_ms.append((time.perf_counter() - t0) * 1000)
        return result

    @property
    def p50(self) -> float:
        ordered = sorted(self.samples_ms)
        return ordered[len(ordered) // 2]

    @property
    def p95(self) -> float:
        ordered = sorted(self.samples_ms)
        idx = min(len(ordered) - 1, int(len(ordered) * 0.95))
        return ordered[idx]

    @property
    def mean(self) -> float:
        return sum(self.samples_ms) / len(self.samples_ms)

    def summary(self) -> str:
        return (
            f"{self.name}: n={len(self.samples_ms)} "
            f"mean={self.mean:.1f}ms p50={self.p50:.1f}ms p95={self.p95:.1f}ms "
            f"max={max(self.samples_ms):.1f}ms"
        )
