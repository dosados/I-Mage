from __future__ import annotations

import uuid
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from db.models import Face, FacePersonAssignment, Image, Person, utc_now_iso
from db.types import FaceAssignmentRecord, PersonFaceRecord, PersonRecord


class PersonService:
    MANUAL_SOURCES = frozenset({"manual_merge", "manual_split", "manual_assign"})

    def __init__(self, session: Session) -> None:
        self._session = session

    def list_persons(
        self,
        *,
        min_face_count: int = 0,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[PersonRecord]:
        face_count = func.count(FacePersonAssignment.face_id).label("face_count")
        stmt = (
            select(Person, face_count)
            .outerjoin(FacePersonAssignment, FacePersonAssignment.person_id == Person.id)
            .group_by(Person.id)
            # Named people first, then largest clusters — the ones worth naming
            # surface at the top instead of drowning under thousands of singletons.
            .order_by(Person.is_named.desc(), face_count.desc(), Person.updated_at.desc())
        )
        if min_face_count > 0:
            stmt = stmt.having(face_count >= min_face_count)
        if offset:
            stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)
        rows = self._session.execute(stmt).all()
        return [
            PersonRecord(
                id=person.id,
                name=person.name,
                is_named=person.is_named,
                face_count=int(count),
                created_at=person.created_at,
                updated_at=person.updated_at,
            )
            for person, count in rows
        ]

    def count_persons(self, *, min_face_count: int = 0) -> int:
        face_count = func.count(FacePersonAssignment.face_id).label("face_count")
        if min_face_count > 0:
            sub = (
                select(Person.id)
                .outerjoin(
                    FacePersonAssignment,
                    FacePersonAssignment.person_id == Person.id,
                )
                .group_by(Person.id)
                .having(face_count >= min_face_count)
                .subquery()
            )
            return int(self._session.scalar(select(func.count()).select_from(sub)) or 0)
        return int(self._session.scalar(select(func.count(Person.id))) or 0)

    def preview_faces_for_persons(
        self, person_ids: list[str], *, per_person: int = 3
    ) -> dict[str, list[PersonFaceRecord]]:
        """Fetch a few representative faces per person in one query (no N+1)."""
        if not person_ids:
            return {}
        result: dict[str, list[PersonFaceRecord]] = {pid: [] for pid in person_ids}
        chunk = 500
        for start in range(0, len(person_ids), chunk):
            batch = person_ids[start : start + chunk]
            stmt = (
                select(Face, Image.path, FacePersonAssignment.person_id)
                .join(FacePersonAssignment, FacePersonAssignment.face_id == Face.id)
                .join(Image, Image.id == Face.image_id)
                .where(FacePersonAssignment.person_id.in_(batch))
                .order_by(Face.detection_score.desc())
            )
            for face, path, person_id in self._session.execute(stmt):
                bucket = result.setdefault(person_id, [])
                if len(bucket) >= per_person:
                    continue
                bucket.append(
                    PersonFaceRecord(
                        face_id=face.id,
                        image_id=face.image_id,
                        image_path=Path(path),
                        bbox=(face.bbox_x1, face.bbox_y1, face.bbox_x2, face.bbox_y2),
                        detection_score=face.detection_score,
                    )
                )
        return result

    def get_person(self, person_id: str) -> PersonRecord | None:
        person = self._session.get(Person, person_id)
        if person is None:
            return None
        face_count = self._session.scalar(
            select(func.count(FacePersonAssignment.face_id)).where(
                FacePersonAssignment.person_id == person_id
            )
        )
        return PersonRecord(
            id=person.id,
            name=person.name,
            is_named=person.is_named,
            face_count=int(face_count or 0),
            created_at=person.created_at,
            updated_at=person.updated_at,
        )

    def create_person(self, *, name: str | None = None, is_named: bool = False) -> PersonRecord:
        now = utc_now_iso()
        person = Person(
            id=str(uuid.uuid4()),
            name=name,
            is_named=is_named,
            created_at=now,
            updated_at=now,
        )
        self._session.add(person)
        self._session.flush()
        return PersonRecord(
            id=person.id,
            name=person.name,
            is_named=person.is_named,
            face_count=0,
            created_at=person.created_at,
            updated_at=person.updated_at,
        )

    def rename_person(self, person_id: str, name: str) -> PersonRecord | None:
        person = self._session.get(Person, person_id)
        if person is None:
            return None
        person.name = name.strip() or None
        person.is_named = person.name is not None
        person.updated_at = utc_now_iso()
        self._session.flush()
        return self.get_person(person_id)

    def merge_persons(self, from_person_id: str, to_person_id: str) -> PersonRecord | None:
        if from_person_id == to_person_id:
            raise ValueError("cannot merge person with itself")

        source = self._session.get(Person, from_person_id)
        target = self._session.get(Person, to_person_id)
        if source is None or target is None:
            return None

        assignments = self._session.scalars(
            select(FacePersonAssignment).where(FacePersonAssignment.person_id == from_person_id)
        ).all()
        now = utc_now_iso()
        for assignment in assignments:
            assignment.person_id = to_person_id
            assignment.source = "manual_merge"
            assignment.assigned_at = now

        self._session.delete(source)
        target.updated_at = now
        self._session.flush()
        return self.get_person(to_person_id)

    def merge_person_ids(self, person_ids: list[str]) -> PersonRecord | None:
        unique: list[str] = []
        seen: set[str] = set()
        for person_id in person_ids:
            if person_id in seen:
                continue
            seen.add(person_id)
            unique.append(person_id)
        if len(unique) < 2:
            raise ValueError("need at least two people to merge")

        records = [self.get_person(person_id) for person_id in unique]
        if any(record is None for record in records):
            return None

        named = [record for record in records if record is not None and record.is_named]
        target_id = named[0].id if named else unique[0]
        for person_id in unique:
            if person_id == target_id:
                continue
            merged = self.merge_persons(person_id, target_id)
            if merged is None:
                return None
        return self.get_person(target_id)

    def split_person(self, person_id: str, face_ids: list[str]) -> PersonRecord | None:
        if not face_ids:
            raise ValueError("face_ids must not be empty")
        created = self.split_person_into_groups(person_id, [face_ids])
        return created[0] if created else None

    def split_person_into_groups(
        self, person_id: str, groups: list[list[str]]
    ) -> list[PersonRecord]:
        person = self._session.get(Person, person_id)
        if person is None:
            return []

        used: set[str] = set()
        cleaned: list[list[str]] = []
        for group in groups:
            ids: list[str] = []
            for face_id in group:
                if not face_id or face_id in used:
                    continue
                assignment = self._session.get(FacePersonAssignment, face_id)
                if assignment is None or assignment.person_id != person_id:
                    continue
                ids.append(face_id)
                used.add(face_id)
            if ids:
                cleaned.append(ids)
        if not cleaned:
            raise ValueError("none of the faces belong to this person")

        now = utc_now_iso()
        created: list[PersonRecord] = []
        for ids in cleaned:
            new_person = self.create_person()
            for face_id in ids:
                assignment = self._session.get(FacePersonAssignment, face_id)
                if assignment is None:
                    continue
                assignment.person_id = new_person.id
                assignment.source = "manual_split"
                assignment.assigned_at = now
            created.append(self.get_person(new_person.id))

        remaining = self._session.scalars(
            select(FacePersonAssignment).where(FacePersonAssignment.person_id == person_id)
        ).all()
        for assignment in remaining:
            assignment.source = "manual_split"
            assignment.assigned_at = now
        person.updated_at = now
        self._session.flush()
        if not remaining and not person.is_named:
            self._session.delete(person)
            self._session.flush()
        return [record for record in created if record is not None]

    def assign_faces(
        self,
        person_id: str,
        face_ids: list[str],
        *,
        source: str = "auto_cluster",
    ) -> None:
        person = self._session.get(Person, person_id)
        if person is None:
            raise ValueError(f"person not found: {person_id}")

        now = utc_now_iso()
        for face_id in face_ids:
            existing = self._session.get(FacePersonAssignment, face_id)
            if existing is not None:
                if existing.source in self.MANUAL_SOURCES:
                    continue
                if existing.person.is_named:
                    continue
                self._session.delete(existing)

            self._session.add(
                FacePersonAssignment(
                    face_id=face_id,
                    person_id=person_id,
                    source=source,
                    assigned_at=now,
                )
            )
        person.updated_at = now
        self._session.flush()

    def clear_auto_assignments(self, face_ids: list[str]) -> None:
        for face_id in face_ids:
            assignment = self._session.get(FacePersonAssignment, face_id)
            if assignment is None:
                continue
            if assignment.source in self.MANUAL_SOURCES:
                continue
            person = assignment.person
            if person.is_named:
                continue
            self._session.delete(assignment)

    def list_unassigned_face_ids(self, face_ids: list[str]) -> list[str]:
        if not face_ids:
            return []
        assigned = set(
            self._session.scalars(
                select(FacePersonAssignment.face_id).where(
                    FacePersonAssignment.face_id.in_(face_ids)
                )
            ).all()
        )
        return [face_id for face_id in face_ids if face_id not in assigned]

    def list_assignments_for_faces(self, face_ids: list[str]) -> list[FaceAssignmentRecord]:
        if not face_ids:
            return []
        rows = self._session.scalars(
            select(FacePersonAssignment).where(FacePersonAssignment.face_id.in_(face_ids))
        ).all()
        return [
            FaceAssignmentRecord(
                face_id=row.face_id,
                person_id=row.person_id,
                source=row.source,
                assigned_at=row.assigned_at,
            )
            for row in rows
        ]

    def restore_assignments(self, assignments: list[FaceAssignmentRecord]) -> None:
        for record in assignments:
            face = self._session.get(Face, record.face_id)
            if face is None:
                continue
            existing = self._session.get(FacePersonAssignment, record.face_id)
            if existing is not None:
                continue
            self._session.add(
                FacePersonAssignment(
                    face_id=record.face_id,
                    person_id=record.person_id,
                    source=record.source,
                    assigned_at=record.assigned_at,
                )
            )

    def list_faces_for_person(self, person_id: str) -> list[PersonFaceRecord]:
        stmt = (
            select(Face, Image.path)
            .join(FacePersonAssignment, FacePersonAssignment.face_id == Face.id)
            .join(Image, Image.id == Face.image_id)
            .where(FacePersonAssignment.person_id == person_id)
            .order_by(Face.detection_score.desc())
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

    def list_regroup_eligible_face_ids(self, face_ids: list[str]) -> list[str]:
        eligible: list[str] = []
        for face_id in face_ids:
            assignment = self._session.get(FacePersonAssignment, face_id)
            if assignment is None:
                eligible.append(face_id)
                continue
            if assignment.source in self.MANUAL_SOURCES:
                continue
            person = self._session.get(Person, assignment.person_id)
            if person is not None and person.is_named:
                continue
            eligible.append(face_id)
        return eligible

    def delete_empty_unnamed_persons(self) -> None:
        persons = self._session.scalars(select(Person).where(Person.is_named.is_(False))).all()
        for person in persons:
            count = self._session.scalar(
                select(func.count(FacePersonAssignment.face_id)).where(
                    FacePersonAssignment.person_id == person.id
                )
            )
            if not count:
                self._session.delete(person)

    def delete_empty_persons(self) -> None:
        """Drop every person with zero faces (named or not).

        After (re)clustering, persons whose faces all moved elsewhere or were
        filtered out are dead rows; removing them keeps the table honest and the
        counts meaningful.
        """
        counts = dict(
            self._session.execute(
                select(
                    FacePersonAssignment.person_id,
                    func.count(FacePersonAssignment.face_id),
                ).group_by(FacePersonAssignment.person_id)
            ).all()
        )
        for person in self._session.scalars(select(Person)).all():
            if not counts.get(person.id):
                self._session.delete(person)
        self._session.flush()
