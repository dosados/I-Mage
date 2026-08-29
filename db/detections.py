from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from db.models import Detection, Image, ImageYolo
from db.types import ClassDetectionMatch, DetectionRecord, ModuleStatus
from ml.objects.base import Detection as MlDetection


class DetectionService:
    def __init__(self, session: Session) -> None:
        self._session = session

    def replace_for_image(self, image_id: str, detections: list[MlDetection]) -> None:
        image = self._session.get(Image, image_id)
        if image is None:
            raise ValueError(f"image not found: {image_id}")

        image.detections.clear()
        image.detections.extend(
            [
                Detection(
                    image_id=image_id,
                    label=detection.label.strip().lower(),
                    confidence=detection.confidence,
                    bbox_x1=detection.bbox[0],
                    bbox_y1=detection.bbox[1],
                    bbox_x2=detection.bbox[2],
                    bbox_y2=detection.bbox[3],
                )
                for detection in detections
            ]
        )

    def list_for_image(self, image_id: str) -> list[DetectionRecord]:
        stmt = (
            select(Detection)
            .where(Detection.image_id == image_id)
            .order_by(Detection.confidence.desc())
        )
        return [row.to_record() for row in self._session.scalars(stmt)]

    def search_by_label(
        self,
        label: str,
        *,
        k: int | None = None,
        paths: set[Path] | None = None,
    ) -> list[ClassDetectionMatch]:
        normalized = label.strip().lower()
        if not normalized:
            raise ValueError("label must not be empty")

        stmt = (
            select(
                Image.path,
                func.max(Detection.confidence).label("confidence"),
            )
            .join(Detection, Detection.image_id == Image.id)
            .join(ImageYolo, ImageYolo.image_id == Image.id)
            .where(
                ImageYolo.status == ModuleStatus.DONE.value,
                Detection.label == normalized,
            )
            .group_by(Image.id)
            .order_by(func.max(Detection.confidence).desc())
        )
        if paths is not None:
            resolved_paths = {str(path.resolve()) for path in paths}
            stmt = stmt.where(Image.path.in_(resolved_paths))
        if k is not None:
            stmt = stmt.limit(k)

        return [
            ClassDetectionMatch(path=Path(path), confidence=confidence)
            for path, confidence in self._session.execute(stmt)
        ]
