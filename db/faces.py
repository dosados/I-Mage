from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from db.face_ids import make_face_id
from db.models import Face, FacePersonAssignment, Image
from db.types import FaceAssignmentRecord, FaceRecord, PersonFaceRecord
from ml.faces.base import Face as MlFace


class FaceService:
    def __init__(self, session: Session) -> None:
        self._session = session

    def replace_for_image(
        self,
        image_id: str,
        faces: list[MlFace],
        *,
        model_version: str,
    ) -> list[FaceRecord]:
        old_face_ids = list(
            self._session.scalars(select(Face.id).where(Face.image_id == image_id))
        )
        saved_assignments: list[FaceAssignmentRecord] = []
        if old_face_ids:
            rows = self._session.scalars(
                select(FacePersonAssignment).where(
                    FacePersonAssignment.face_id.in_(old_face_ids)
                )
            ).all()
            saved_assignments = [
                FaceAssignmentRecord(
                    face_id=row.face_id,
                    person_id=row.person_id,
                    source=row.source,
                    assigned_at=row.assigned_at,
                )
                for row in rows
            ]

        self._session.execute(delete(Face).where(Face.image_id == image_id))

        rows: list[Face] = []
        for face in faces:
            row = Face(
                id=make_face_id(image_id, face.bbox, model_version=model_version),
                image_id=image_id,
                bbox_x1=face.bbox[0],
                bbox_y1=face.bbox[1],
                bbox_x2=face.bbox[2],
                bbox_y2=face.bbox[3],
                detection_score=face.detection_score,
            )
            self._session.add(row)
            rows.append(row)

        self._session.flush()

        for record in saved_assignments:
            if self._session.get(Face, record.face_id) is None:
                continue
            if self._session.get(FacePersonAssignment, record.face_id) is not None:
                continue
            self._session.add(
                FacePersonAssignment(
                    face_id=record.face_id,
                    person_id=record.person_id,
                    source=record.source,
                    assigned_at=record.assigned_at,
                )
            )
        self._session.flush()

        return [face.to_record() for face in rows]

    def list_for_image(self, image_id: str) -> list[FaceRecord]:
        stmt = (
            select(Face)
            .where(Face.image_id == image_id)
            .order_by(Face.detection_score.desc())
        )
        return [face.to_record() for face in self._session.scalars(stmt)]

    def list_ids_in_scope(self, paths: set[Path]) -> list[str]:
        if not paths:
            return []
        resolved = {str(path.resolve()) for path in paths}
        stmt = (
            select(Face.id)
            .join(Image, Image.id == Face.image_id)
            .where(Image.path.in_(resolved))
        )
        return list(self._session.scalars(stmt))

    def count_low_confidence(self, max_score: float) -> int:
        from sqlalchemy import func

        return int(
            self._session.scalar(
                select(func.count(Face.id)).where(Face.detection_score < max_score)
            )
            or 0
        )

    def list_low_confidence(
        self, max_score: float, *, limit: int = 200
    ) -> list[PersonFaceRecord]:
        """Faces below the clustering confidence floor — kept out of people groups.

        Surfaced separately for debugging; later they'll be dropped entirely.
        """
        stmt = (
            select(Face, Image.path)
            .join(Image, Image.id == Face.image_id)
            .where(Face.detection_score < max_score)
            .order_by(Face.detection_score.asc())
            .limit(limit)
        )
        return [
            PersonFaceRecord(
                face_id=face.id,
                image_id=face.image_id,
                image_path=Path(path),
                bbox=(face.bbox_x1, face.bbox_y1, face.bbox_x2, face.bbox_y2),
                detection_score=face.detection_score,
            )
            for face, path in self._session.execute(stmt)
        ]

    def list_confident_ids(self, face_ids: list[str], min_score: float) -> list[str]:
        """Return the subset of face_ids whose detection_score >= min_score.

        Low-confidence detections (blurry/partial/false-positive faces) produce
        unreliable embeddings that collapse into a single huge "junk" cluster and
        chain unrelated people together; filtering them out keeps clusters clean.
        """
        if not face_ids:
            return []
        result: list[str] = []
        chunk = 900  # stay under SQLite's variable limit
        for start in range(0, len(face_ids), chunk):
            batch = face_ids[start : start + chunk]
            stmt = select(Face.id).where(
                Face.id.in_(batch), Face.detection_score >= min_score
            )
            result.extend(self._session.scalars(stmt))
        return result
