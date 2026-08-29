from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from db.database import Database
from ml.faces.base import Face as MlFace
from ml.objects.base import Detection as MlDetection
from helpers import make_image_file, register_file


def _image(db: Database, image_dir: Path, name: str = "a.jpg"):
    path = make_image_file(image_dir, name)
    return register_file(db, path), path


class TestDetections:
    def test_replace_and_list(self, db: Database, image_dir: Path) -> None:
        with db:
            record, _ = _image(db, image_dir)
            db.image_yolo.mark_done(record.id, model_version="y")
            db.detections.replace_for_image(
                record.id,
                [
                    MlDetection(label="dog", confidence=0.9, bbox=(0, 0, 1, 1)),
                    MlDetection(label="cat", confidence=0.5, bbox=(1, 1, 2, 2)),
                ],
            )
            rows = db.detections.list_for_image(record.id)
        assert [r.label for r in rows] == ["dog", "cat"]
        assert rows[0].confidence == 0.9

    def test_replace_unknown_image_raises(self, db: Database) -> None:
        with db:
            with pytest.raises(ValueError, match="image not found"):
                db.detections.replace_for_image("missing", [])

    def test_search_by_label(self, db: Database, image_dir: Path) -> None:
        with db:
            a, path_a = _image(db, image_dir, "a.jpg")
            b, path_b = _image(db, image_dir, "b.jpg")
            db.image_yolo.mark_done(a.id, model_version="y")
            db.image_yolo.mark_done(b.id, model_version="y")
            db.detections.replace_for_image(
                a.id, [MlDetection(label="Dog", confidence=0.8, bbox=(0, 0, 1, 1))]
            )
            db.detections.replace_for_image(
                b.id, [MlDetection(label="dog", confidence=0.95, bbox=(0, 0, 1, 1))]
            )
            matches = db.detections.search_by_label("dog", k=10)
        assert [m.path for m in matches] == [path_b, path_a]
        assert matches[0].confidence == 0.95

    def test_search_by_label_empty_raises(self, db: Database) -> None:
        with db:
            with pytest.raises(ValueError, match="label must not be empty"):
                db.detections.search_by_label("   ")


class TestFaces:
    def test_replace_preserves_manual_assignments(
        self, db: Database, image_dir: Path
    ) -> None:
        with db:
            record, path = _image(db, image_dir)
            emb = np.zeros(4, dtype=np.float32)
            faces = db.faces.replace_for_image(
                record.id,
                [MlFace(bbox=(0, 0, 10, 10), detection_score=0.99, embedding=emb)],
                model_version="v1",
            )
            assert len(faces) == 1
            face_id = faces[0].id
            person = db.persons.create_person(name="Ada", is_named=True)
            db.persons.assign_faces(person.id, [face_id], source="manual_assign")

            # Replacing with same bbox/model keeps face_id stable → assignment survives.
            again = db.faces.replace_for_image(
                record.id,
                [MlFace(bbox=(0, 0, 10, 10), detection_score=0.99, embedding=emb)],
                model_version="v1",
            )
            assert again[0].id == face_id
            assignments = db.persons.list_assignments_for_faces([face_id])
            assert len(assignments) == 1
            assert assignments[0].person_id == person.id

            listed = db.faces.list_for_image(record.id)
            assert listed[0].id == face_id
            assert db.faces.list_ids_in_scope({path}) == [face_id]

    def test_low_confidence_helpers(self, db: Database, image_dir: Path) -> None:
        with db:
            record, _ = _image(db, image_dir)
            emb = np.zeros(4, dtype=np.float32)
            faces = db.faces.replace_for_image(
                record.id,
                [
                    MlFace(bbox=(0, 0, 1, 1), detection_score=0.2, embedding=emb),
                    MlFace(bbox=(1, 1, 2, 2), detection_score=0.9, embedding=emb),
                ],
                model_version="v1",
            )
            assert db.faces.count_low_confidence(0.5) == 1
            low = db.faces.list_low_confidence(0.5, limit=10)
            assert len(low) == 1
            confident = db.faces.list_confident_ids(
                [f.id for f in faces], min_score=0.5
            )
            assert len(confident) == 1


class TestPersons:
    def test_create_rename_list(self, db: Database, image_dir: Path) -> None:
        with db:
            record, _ = _image(db, image_dir)
            emb = np.zeros(4, dtype=np.float32)
            faces = db.faces.replace_for_image(
                record.id,
                [MlFace(bbox=(0, 0, 1, 1), detection_score=0.9, embedding=emb)],
                model_version="v1",
            )
            person = db.persons.create_person()
            db.persons.assign_faces(person.id, [faces[0].id])
            renamed = db.persons.rename_person(person.id, "Bob")
            assert renamed is not None
            assert renamed.name == "Bob"
            assert renamed.is_named is True
            people = db.persons.list_persons(min_face_count=1)
            assert len(people) == 1
            assert people[0].face_count == 1
            assert db.persons.count_persons(min_face_count=1) == 1

    def test_merge_and_split(self, db: Database, image_dir: Path) -> None:
        with db:
            record, _ = _image(db, image_dir)
            emb = np.zeros(4, dtype=np.float32)
            faces = db.faces.replace_for_image(
                record.id,
                [
                    MlFace(bbox=(0, 0, 1, 1), detection_score=0.9, embedding=emb),
                    MlFace(bbox=(2, 2, 3, 3), detection_score=0.8, embedding=emb),
                ],
                model_version="v1",
            )
            a = db.persons.create_person(name="A", is_named=True)
            b = db.persons.create_person(name="B", is_named=True)
            db.persons.assign_faces(a.id, [faces[0].id], source="manual_assign")
            db.persons.assign_faces(b.id, [faces[1].id], source="manual_assign")

            merged = db.persons.merge_persons(b.id, a.id)
            assert merged is not None
            assert merged.id == a.id
            assert db.persons.get_person(b.id) is None
            assert len(db.persons.list_faces_for_person(a.id)) == 2

            split = db.persons.split_person(a.id, [faces[1].id])
            assert split is not None
            assert split.id != a.id
            assert len(db.persons.list_faces_for_person(a.id)) == 1
            assert len(db.persons.list_faces_for_person(split.id)) == 1
            sources = {
                row.source
                for row in db.persons.list_assignments_for_faces([faces[0].id, faces[1].id])
            }
            assert sources == {"manual_split"}

    def test_merge_person_ids_prefers_named(self, db: Database, image_dir: Path) -> None:
        with db:
            record, _ = _image(db, image_dir)
            emb = np.zeros(4, dtype=np.float32)
            faces = db.faces.replace_for_image(
                record.id,
                [
                    MlFace(bbox=(0, 0, 1, 1), detection_score=0.9, embedding=emb),
                    MlFace(bbox=(2, 2, 3, 3), detection_score=0.8, embedding=emb),
                    MlFace(bbox=(4, 4, 5, 5), detection_score=0.7, embedding=emb),
                ],
                model_version="v1",
            )
            unnamed_a = db.persons.create_person()
            named = db.persons.create_person(name="Ada", is_named=True)
            unnamed_b = db.persons.create_person()
            db.persons.assign_faces(unnamed_a.id, [faces[0].id], source="auto_cluster")
            db.persons.assign_faces(named.id, [faces[1].id], source="auto_cluster")
            db.persons.assign_faces(unnamed_b.id, [faces[2].id], source="auto_cluster")
            merged = db.persons.merge_person_ids([unnamed_a.id, named.id, unnamed_b.id])
            assert merged is not None
            assert merged.id == named.id
            assert merged.face_count == 3
            assert db.persons.get_person(unnamed_a.id) is None
            assert db.persons.get_person(unnamed_b.id) is None

    def test_split_into_groups_freezes_remaining(self, db: Database, image_dir: Path) -> None:
        with db:
            record, _ = _image(db, image_dir)
            emb = np.zeros(4, dtype=np.float32)
            faces = db.faces.replace_for_image(
                record.id,
                [
                    MlFace(bbox=(0, 0, 1, 1), detection_score=0.9, embedding=emb),
                    MlFace(bbox=(2, 2, 3, 3), detection_score=0.8, embedding=emb),
                    MlFace(bbox=(4, 4, 5, 5), detection_score=0.7, embedding=emb),
                ],
                model_version="v1",
            )
            person = db.persons.create_person()
            ids = [face.id for face in faces]
            db.persons.assign_faces(person.id, ids, source="auto_cluster")
            created = db.persons.split_person_into_groups(person.id, [[ids[0]], [ids[1]]])
            assert len(created) == 2
            assert len(db.persons.list_faces_for_person(person.id)) == 1
            assignments = db.persons.list_assignments_for_faces(ids)
            assert {row.source for row in assignments} == {"manual_split"}
            assert db.persons.list_regroup_eligible_face_ids(ids) == []

    def test_merge_self_raises(self, db: Database) -> None:
        with db:
            person = db.persons.create_person()
            with pytest.raises(ValueError, match="cannot merge"):
                db.persons.merge_persons(person.id, person.id)

    def test_clear_auto_keeps_manual_and_named(self, db: Database, image_dir: Path) -> None:
        with db:
            record, _ = _image(db, image_dir)
            emb = np.zeros(4, dtype=np.float32)
            faces = db.faces.replace_for_image(
                record.id,
                [
                    MlFace(bbox=(0, 0, 1, 1), detection_score=0.9, embedding=emb),
                    MlFace(bbox=(2, 2, 3, 3), detection_score=0.8, embedding=emb),
                ],
                model_version="v1",
            )
            auto_person = db.persons.create_person()
            named = db.persons.create_person(name="Named", is_named=True)
            db.persons.assign_faces(auto_person.id, [faces[0].id], source="auto_cluster")
            db.persons.assign_faces(named.id, [faces[1].id], source="manual_assign")
            db.persons.clear_auto_assignments([faces[0].id, faces[1].id])
            assert db.persons.list_assignments_for_faces([faces[0].id]) == []
            assert len(db.persons.list_assignments_for_faces([faces[1].id])) == 1

    def test_preview_faces_for_persons(self, db: Database, image_dir: Path) -> None:
        with db:
            record, _ = _image(db, image_dir)
            emb = np.zeros(4, dtype=np.float32)
            faces = db.faces.replace_for_image(
                record.id,
                [
                    MlFace(bbox=(0, 0, 1, 1), detection_score=0.9, embedding=emb),
                    MlFace(bbox=(2, 2, 3, 3), detection_score=0.7, embedding=emb),
                ],
                model_version="v1",
            )
            person = db.persons.create_person(name="P", is_named=True)
            db.persons.assign_faces(
                person.id, [f.id for f in faces], source="manual_assign"
            )
            preview = db.persons.preview_faces_for_persons([person.id], per_person=1)
        assert len(preview[person.id]) == 1
        assert preview[person.id][0].detection_score == 0.9

    def test_delete_empty_persons(self, db: Database) -> None:
        with db:
            empty = db.persons.create_person(name="Ghost", is_named=True)
            db.persons.delete_empty_persons()
            assert db.persons.get_person(empty.id) is None
