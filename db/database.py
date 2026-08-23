from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from db.detections import DetectionService
from db.engine import create_engine_for_path, create_session_factory, init_database
from db.faces import FaceService
from db.hash import compute_content_hash, read_file_stat
from db.image_clip import ImageClipService
from db.image_faces import ImageFacesService
from db.image_yolo import ImageYoloService
from db.images import ImageService
from db.index_runs import IndexRunService
from db.persons import PersonService
from db.scan_config import ScanConfig, ScanConfigService, default_scan_config
from db.types import ImageRecord, ReconcileResult

if TYPE_CHECKING:
    from vectors.store import VectorStore


class _ThreadState:
    __slots__ = (
        "session",
        "images",
        "image_yolo",
        "image_clip",
        "image_faces",
        "detections",
        "faces",
        "scan_config",
        "index_runs",
        "persons",
    )

    def __init__(self) -> None:
        self.session: Session | None = None
        self.images: ImageService | None = None
        self.image_yolo: ImageYoloService | None = None
        self.image_clip: ImageClipService | None = None
        self.image_faces: ImageFacesService | None = None
        self.detections: DetectionService | None = None
        self.faces: FaceService | None = None
        self.scan_config: ScanConfigService | None = None
        self.index_runs: IndexRunService | None = None
        self.persons: PersonService | None = None


class Database:
    """Facade over SQLAlchemy services used by indexing and search."""

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        vector_store: VectorStore | None = None,
    ) -> None:
        self._vector_store = vector_store
        self.engine = create_engine_for_path(path)
        self.path = Path(self.engine.url.database or "")
        init_database(self.engine)
        self._session_factory = create_session_factory(self.engine)
        self._local = threading.local()
        self._default_scan_config: ScanConfig | None = None

    def _state(self) -> _ThreadState:
        state = getattr(self._local, "state", None)
        if state is None:
            state = _ThreadState()
            self._local.state = state
        return state

    def _clear_services(self, state: _ThreadState) -> None:
        state.images = None
        state.image_yolo = None
        state.image_clip = None
        state.image_faces = None
        state.detections = None
        state.faces = None
        state.scan_config = None
        state.index_runs = None
        state.persons = None

    @property
    def session(self) -> Session:
        state = self._state()
        if state.session is None:
            state.session = self._session_factory()
        return state.session

    @property
    def images(self) -> ImageService:
        state = self._state()
        if state.images is None:
            state.images = ImageService(self.session, vector_store=self._vector_store)
        return state.images

    @property
    def image_yolo(self) -> ImageYoloService:
        state = self._state()
        if state.image_yolo is None:
            state.image_yolo = ImageYoloService(self.session)
        return state.image_yolo

    @property
    def image_clip(self) -> ImageClipService:
        state = self._state()
        if state.image_clip is None:
            state.image_clip = ImageClipService(self.session)
        return state.image_clip

    @property
    def image_faces(self) -> ImageFacesService:
        state = self._state()
        if state.image_faces is None:
            state.image_faces = ImageFacesService(self.session)
        return state.image_faces

    @property
    def detections(self) -> DetectionService:
        state = self._state()
        if state.detections is None:
            state.detections = DetectionService(self.session)
        return state.detections

    @property
    def faces(self) -> FaceService:
        state = self._state()
        if state.faces is None:
            state.faces = FaceService(self.session)
        return state.faces

    @property
    def scan_config(self) -> ScanConfigService:
        state = self._state()
        if state.scan_config is None:
            state.scan_config = ScanConfigService(self.session)
        return state.scan_config

    @property
    def index_runs(self) -> IndexRunService:
        state = self._state()
        if state.index_runs is None:
            state.index_runs = IndexRunService(self.session)
        return state.index_runs

    @property
    def persons(self) -> PersonService:
        state = self._state()
        if state.persons is None:
            state.persons = PersonService(self.session)
        return state.persons

    def set_default_scan_config(self, config: ScanConfig) -> None:
        self._default_scan_config = config

    def get_scan_config(self) -> ScanConfig:
        defaults = self._default_scan_config or ScanConfig()
        return self.scan_config.get(defaults=defaults)

    def register_image_file(self, path: Path) -> ImageRecord:
        content_hash = compute_content_hash(path)
        mtime, size = read_file_stat(path)
        return self.images.upsert_from_file(
            path,
            content_hash=content_hash,
            mtime=mtime,
            size=size,
        )

    def reconcile_paths(
        self,
        paths: set[Path],
        *,
        remove_missing: bool = False,
        on_progress: Callable[[int, int], None] | None = None,
        commit_batch_size: int = 250,
    ) -> ReconcileResult:
        return self.images.reconcile_paths(
            paths,
            remove_missing=remove_missing,
            on_progress=on_progress,
            commit_batch_size=commit_batch_size,
        )

    def commit(self) -> None:
        state = self._state()
        if state.session is not None:
            state.session.commit()

    def rollback(self) -> None:
        state = self._state()
        if state.session is not None:
            state.session.rollback()

    def close(self) -> None:
        state = self._state()
        if state.session is not None:
            state.session.close()
            state.session = None
        self._clear_services(state)

    def __enter__(self) -> "Database":
        self.session
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        self.close()
