from sqlalchemy.orm import Session

from db.models import ImageYolo, utc_now_iso
from db.types import ModuleIndex, ModuleStatus


class ImageYoloService:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, image_id: str) -> ModuleIndex | None:
        row = self._session.get(ImageYolo, image_id)
        return row.to_module_index() if row is not None else None

    def is_indexed(self, image_id: str) -> bool:
        row = self._session.get(ImageYolo, image_id)
        return row is not None and ModuleStatus(row.status) == ModuleStatus.DONE

    def is_running(self, image_id: str) -> bool:
        row = self._session.get(ImageYolo, image_id)
        return row is not None and ModuleStatus(row.status) == ModuleStatus.RUNNING

    def needs_reindex(self, image_id: str) -> bool:
        row = self._session.get(ImageYolo, image_id)
        if row is None:
            return True
        return ModuleStatus(row.status) == ModuleStatus.FAILED

    def mark_running(self, image_id: str, *, model_version: str = "") -> ModuleIndex:
        return self._set_status(
            image_id,
            ModuleStatus.RUNNING,
            model_version=model_version,
            last_error=None,
        )

    def mark_done(self, image_id: str, *, model_version: str = "") -> ModuleIndex:
        return self._set_status(
            image_id,
            ModuleStatus.DONE,
            model_version=model_version,
            last_error=None,
        )

    def mark_failed(
        self,
        image_id: str,
        error: str,
        *,
        model_version: str = "",
    ) -> ModuleIndex:
        return self._set_status(
            image_id,
            ModuleStatus.FAILED,
            model_version=model_version,
            last_error=error,
        )

    def _set_status(
        self,
        image_id: str,
        status: ModuleStatus,
        *,
        model_version: str,
        last_error: str | None,
    ) -> ModuleIndex:
        row = self._session.get(ImageYolo, image_id)
        if row is None:
            row = ImageYolo(image_id=image_id)
            self._session.add(row)

        row.status = status.value
        row.model_version = model_version
        row.last_error = last_error

        if status == ModuleStatus.DONE:
            row.indexed_at = utc_now_iso()
        else:
            row.indexed_at = None

        self._session.flush()
        return row.to_module_index()
