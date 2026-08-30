from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from api.search import (
    ClassSearchMatch,
    _merge_unified_matches,
    iter_face_search_events,
    run_embed_query_face,
    run_search_by_class,
    run_search_by_description,
    run_search_by_face,
)
from db.database import Database
from db.types import MODULE_FACES
from fakes import FakeClip, FakeFaces, FakeVectorStore, FakeYolo, unit_vec
from helpers import configure_scan, make_image_file, register_file, seed_catalog
from indexing.clip import index_clip_image
from indexing.faces import index_faces_image
from indexing.yolo import index_yolo_image
from ml.faces.base import Face
from ml.faces.service import encode_query_face, search_by_face
from ml.objects.base import Detection
from PIL import Image
from ml.embeddings.service import ImageMatch


def _rgb_file(directory: Path, name: str, color: tuple[int, int, int] = (10, 20, 30)) -> Path:
    path = directory / name
    Image.new("RGB", (32, 32), color).save(path)
    return path.resolve()


class TestFaceEmbedAndBruteforce:
    def test_encode_query_face_picks_best(self) -> None:
        faces = FakeFaces()
        faces.set_faces(
            "upload",
            [
                Face(bbox=(0, 0, 1, 1), detection_score=0.5, embedding=unit_vec(1)),
                Face(bbox=(0, 0, 2, 2), detection_score=0.95, embedding=unit_vec(2)),
            ],
        )
        # analyze key for non-path uses filename attribute — use Path key
        faces.set_faces(
            "query.jpg",
            [
                Face(bbox=(0, 0, 1, 1), detection_score=0.5, embedding=unit_vec(1)),
                Face(bbox=(0, 0, 2, 2), detection_score=0.95, embedding=unit_vec(2)),
            ],
        )
        result = encode_query_face("query.jpg", faces)
        assert np.allclose(result.embedding, unit_vec(2))
        assert result.detection_score == 0.95

    def test_encode_query_no_face_raises(self) -> None:
        faces = FakeFaces()
        faces.set_faces("empty.jpg", [])
        with pytest.raises(ValueError, match="no face found"):
            encode_query_face("empty.jpg", faces)

    def test_bruteforce_search_by_face(self, image_dir: Path) -> None:
        target = unit_vec(42)
        a = make_image_file(image_dir, "a.jpg")
        b = make_image_file(image_dir, "b.jpg")
        faces = FakeFaces()
        faces.set_faces(
            "a.jpg",
            [Face(bbox=(0, 0, 1, 1), detection_score=0.9, embedding=target)],
        )
        faces.set_faces(
            "b.jpg",
            [Face(bbox=(0, 0, 1, 1), detection_score=0.9, embedding=unit_vec(7))],
        )
        result = search_by_face(target, [a, b], faces, k=5, threshold=0.5)
        assert result.matches[0].path == a


class TestSearchByFaceIndexed:
    def test_indexed_search_does_not_reanalyze_catalog(
        self, db: Database, image_dir: Path
    ) -> None:
        paths = [_rgb_file(image_dir, f"{i}.jpg", (i * 10, 0, 0)) for i in range(5)]
        configure_scan(db, [image_dir])
        store = FakeVectorStore()
        faces = FakeFaces()
        query = unit_vec(99)

        with db:
            for path in paths:
                register_file(db, path)
                # Make one face match the query closely.
                if path.name == "2.jpg":
                    faces.set_faces(
                        path.name,
                        [Face(bbox=(0, 0, 5, 5), detection_score=0.99, embedding=query)],
                    )
                index_faces_image(db, store, path, faces, model_version=faces.model_name)

        faces.analyze_calls.clear()
        result = run_search_by_face(
            query,
            faces,
            db,
            store,
            k=3,
            threshold=0.1,
        )
        assert faces.analyze_calls == []  # no catalog re-scan
        assert store.search_faces_calls == 1
        assert result.matches
        assert result.matches[0].path.name == "2.jpg"

    def test_empty_matches_not_404_when_index_exists(
        self, db: Database, image_dir: Path
    ) -> None:
        path = _rgb_file(image_dir, "only.jpg")
        configure_scan(db, [image_dir])
        store = FakeVectorStore()
        faces = FakeFaces()
        with db:
            register_file(db, path)
            index_faces_image(db, store, path, faces, model_version=faces.model_name)

        faces.analyze_calls.clear()
        # Orthogonal query → below threshold, but still a successful empty result.
        result = run_search_by_face(
            unit_vec(12345),
            faces,
            db,
            store,
            k=5,
            threshold=0.99,
        )
        assert result.matches == []
        assert faces.analyze_calls == []  # still no bruteforce

    def test_qdrant_down_raises_without_bruteforce(
        self, db: Database, image_dir: Path
    ) -> None:
        make_image_file(image_dir, "a.jpg")
        configure_scan(db, [image_dir])
        with db:
            register_file(db, image_dir / "a.jpg")
        with pytest.raises(ValueError, match="Qdrant unavailable"):
            run_search_by_face(
                unit_vec(1),
                FakeFaces(),
                db,
                FakeVectorStore(available=False),
                allow_bruteforce_fallback=False,
            )

    def test_gap_is_not_indexed_by_search(
        self, db: Database, image_dir: Path
    ) -> None:
        paths = [_rgb_file(image_dir, f"{i}.jpg") for i in range(3)]
        configure_scan(db, [image_dir])
        store = FakeVectorStore()
        faces = FakeFaces()
        with db:
            for path in paths:
                register_file(db, path)
            # Only first two indexed — third is a gap.
            for path in paths[:2]:
                index_faces_image(db, store, path, faces, model_version=faces.model_name)

        faces.analyze_calls.clear()
        result = run_search_by_face(unit_vec(0), faces, db, store, k=5, threshold=-1.0)
        assert faces.analyze_calls == []
        assert result.matches
        assert db.images.count_module_done(MODULE_FACES) == 2

    def test_stream_events_order(self, db: Database, image_dir: Path) -> None:
        path = _rgb_file(image_dir, "a.jpg")
        configure_scan(db, [image_dir])
        store = FakeVectorStore()
        faces = FakeFaces()
        with db:
            register_file(db, path)
            index_faces_image(db, store, path, faces, model_version=faces.model_name)

        events = list(
            iter_face_search_events(unit_vec(0), faces, db, store, k=3, threshold=0.0)
        )
        stages = [(e["stage"], e["status"]) for e in events]
        assert stages[0] == ("search", "running")
        assert ("search", "running") in stages
        assert stages[-1][0] == "done"
        assert "matches" in events[-1]


class TestSearchDescriptionAndClass:

    def test_unified_ranking_prioritizes_object_overlap_then_clip_score(self) -> None:
        both = Path("both.jpg")
        cat_with_context = Path("cat-with-context.jpg")
        dog_only = Path("dog-only.jpg")
        context_only = Path("context-only.jpg")

        matches = _merge_unified_matches(
            [
                ImageMatch(path=both, score=0.20),
                ImageMatch(path=cat_with_context, score=0.95),
                ImageMatch(path=dog_only, score=0.10),
                ImageMatch(path=context_only, score=0.99),
            ],
            {
                "cat": [
                    ClassSearchMatch(path=both, confidence=0.50),
                    ClassSearchMatch(path=cat_with_context, confidence=0.99),
                ],
                "dog": [
                    ClassSearchMatch(path=both, confidence=0.40),
                    ClassSearchMatch(path=dog_only, confidence=0.98),
                ],
            },
            k=10,
        )

        assert [match.path for match in matches] == [
            both,
            cat_with_context,
            dog_only,
            context_only,
        ]

    def test_description_uses_vector_index(
        self, db: Database, image_dir: Path
    ) -> None:
        paths = [_rgb_file(image_dir, f"{i}.jpg") for i in range(4)]
        configure_scan(db, [image_dir])
        store = FakeVectorStore()
        clip = FakeClip()
        with db:
            for path in paths:
                register_file(db, path)
                index_clip_image(db, store, path, clip, model_version=clip.model_name)

        clip.encode_image_calls.clear()
        progress: list[tuple[str, str]] = []
        result = run_search_by_description(
            "cat",
            clip,
            db,
            store,
            k=2,
            on_progress=lambda stage, status, payload: progress.append((stage, status)),
        )
        assert clip.encode_image_calls == []
        assert clip.encode_text_calls == ["cat"]
        assert store.search_context_calls == 1
        assert len(result.matches) <= 2
        assert ("catalog", "done") in progress
        assert ("search", "done") in progress

    def test_class_search_only_queries_existing_index(
        self, db: Database, image_dir: Path
    ) -> None:
        path = _rgb_file(image_dir, "pet.jpg")
        configure_scan(db, [image_dir])
        yolo = FakeYolo()
        yolo.set_detections(
            path,
            [Detection(label="cat", confidence=0.91, bbox=(0, 0, 8, 8))],
        )
        with db:
            register_file(db, path)
            index_yolo_image(db, path, yolo, model_version=yolo.model_name)

        yolo.detect_calls.clear()
        result = run_search_by_class("cat", yolo, db, k=5)
        assert result.matches[0].path == path
        assert result.matches[0].confidence == 0.91
        assert yolo.detect_calls == []

        again = run_search_by_class("cat", yolo, db, k=5)
        assert again.matches
        assert yolo.detect_calls == []

    def test_class_missing_returns_empty(self, db: Database, image_dir: Path) -> None:
        path = _rgb_file(image_dir, "x.jpg")
        configure_scan(db, [image_dir])
        with db:
            register_file(db, path)
            index_yolo_image(db, path, FakeYolo(), model_version="y")
        result = run_search_by_class("unicorn", FakeYolo(), db)
        assert result.matches == []

    def test_description_stream_events(self, db: Database, image_dir: Path) -> None:
        from api.search import iter_description_search_events
        from indexing.clip import index_clip_image

        path = _rgb_file(image_dir, "a.jpg")
        configure_scan(db, [image_dir])
        store = FakeVectorStore()
        clip = FakeClip()
        with db:
            register_file(db, path)
            index_clip_image(db, store, path, clip, model_version=clip.model_name)
        events = list(
            iter_description_search_events("cat", clip, db, store, k=3)
        )
        stages = [e["stage"] for e in events]
        assert stages[0] == "search"
        assert "done" in stages
        assert events[-1]["stage"] == "done"

    def test_class_stream_events(self, db: Database, image_dir: Path) -> None:
        from api.search import iter_class_search_events

        path = _rgb_file(image_dir, "dog.jpg")
        configure_scan(db, [image_dir])
        yolo = FakeYolo()
        yolo.set_detections(
            path,
            [Detection(label="dog", confidence=0.9, bbox=(0, 0, 1, 1))],
        )
        with db:
            register_file(db, path)
            index_yolo_image(db, path, yolo, model_version=yolo.model_name)
        yolo.detect_calls.clear()
        events = list(iter_class_search_events("dog", yolo, db, k=5))
        assert events[-1]["stage"] == "done"
        assert events[-1]["matches"]
        assert yolo.detect_calls == []

        path = _rgb_file(image_dir, "face.jpg")
        faces = FakeFaces()
        faces.set_faces(
            "face.jpg",
            [Face(bbox=(0, 0, 1, 1), detection_score=0.8, embedding=unit_vec(3))],
        )
        result = run_embed_query_face(path, faces)
        assert result.detection_score == 0.8
