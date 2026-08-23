from __future__ import annotations

import numpy as np

from db.face_ids import make_face_id
from fakes import FakeVectorStore, unit_vec
from vectors.ids import context_point_id, face_point_id


class TestIds:
    def test_face_id_stable_and_sensitive(self) -> None:
        a = make_face_id("img", (0, 0, 10, 10), model_version="v1")
        b = make_face_id("img", (0, 0, 10, 10), model_version="v1")
        c = make_face_id("img", (0, 0, 10, 11), model_version="v1")
        d = make_face_id("img", (0, 0, 10, 10), model_version="v2")
        assert a == b
        assert a != c
        assert a != d

    def test_vector_point_ids_stable(self) -> None:
        assert context_point_id("i1", "clip") == context_point_id("i1", "clip")
        assert context_point_id("i1", "clip") != context_point_id("i1", "other")
        assert face_point_id("f1") == face_point_id("f1")
        assert face_point_id("f1") != face_point_id("f2")


class TestFakeVectorStoreSemantics:
    """Behavioural contract that real VectorStore must satisfy (via Fake)."""

    def test_context_search_ranks_by_similarity(self) -> None:
        store = FakeVectorStore()
        target = unit_vec(1)
        store.upsert_context("near", target, model_version="m")
        store.upsert_context("far", unit_vec(999), model_version="m")
        hits = store.search_context(target, ["near", "far"], k=2)
        assert hits[0].image_id == "near"
        assert hits[0].score > hits[1].score

    def test_face_search_respects_threshold_and_scope(self) -> None:
        store = FakeVectorStore()
        q = unit_vec(5)
        store.upsert_faces(
            [
                ("f1", "img-a", q),
                ("f2", "img-b", unit_vec(6)),
                ("f3", "img-c", q),
            ],
            model_version="m",
        )
        hits = store.search_faces(q, ["img-a", "img-b"], limit=10, score_threshold=0.9)
        assert [h.face_id for h in hits] == ["f1"]

    def test_delete_for_image_removes_both_collections(self) -> None:
        store = FakeVectorStore()
        store.upsert_context("img", unit_vec(1), model_version="m")
        store.upsert_faces([("f", "img", unit_vec(2))], model_version="m")
        store.delete_for_image("img")
        assert store.context == {}
        assert store.faces == {}

    def test_large_scope_without_filter_still_returns_hits(self) -> None:
        """Mirrors production: >1000 ids → no MatchAny, post-filter in Python."""
        store = FakeVectorStore()
        q = unit_vec(3)
        ids = [f"img-{i}" for i in range(1500)]
        store.upsert_faces(
            [(f"f-{i}", ids[i], q if i == 42 else unit_vec(i + 10)) for i in range(1500)],
            model_version="m",
        )
        # Fake always post-filters; ensure scoped search finds the needle.
        hits = store.search_faces(q, ids, limit=5, score_threshold=0.5)
        assert any(h.image_id == "img-42" for h in hits)
