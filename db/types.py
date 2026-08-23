from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

MODULE_YOLO = "yolo"
MODULE_CLIP = "clip"
MODULE_FACES = "faces"


class ModuleStatus(str, Enum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True)
class ModuleIndex:
    status: ModuleStatus
    model_version: str
    indexed_at: str | None
    last_error: str | None


@dataclass(frozen=True)
class ImageRecord:
    id: str
    path: Path
    content_hash: str
    mtime: float
    size: int
    created_at: str
    updated_at: str
    modules: dict[str, ModuleIndex] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectionRecord:
    id: int
    image_id: str
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class ClassDetectionMatch:
    path: Path
    confidence: float


@dataclass(frozen=True)
class FaceRecord:
    id: str
    image_id: str
    bbox: tuple[float, float, float, float]
    detection_score: float


@dataclass(frozen=True)
class ReconcileResult:
    upserted: int
    removed: int


@dataclass(frozen=True)
class IndexRunRecord:
    id: str
    module: str
    mode: str
    status: str
    progress_done: int
    progress_total: int
    started_at: str
    finished_at: str | None
    last_error: str | None
    phase: str = "indexing"


@dataclass(frozen=True)
class PersonRecord:
    id: str
    name: str | None
    is_named: bool
    face_count: int
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class FaceAssignmentRecord:
    face_id: str
    person_id: str
    source: str
    assigned_at: str


@dataclass(frozen=True)
class PersonFaceRecord:
    face_id: str
    image_id: str
    image_path: Path
    bbox: tuple[float, float, float, float]
    detection_score: float
