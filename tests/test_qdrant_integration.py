from __future__ import annotations

import os
import uuid

import httpx
import numpy as np
import pytest

from vectors.config import CLIP_VECTOR_DIM, FACE_VECTOR_DIM
from vectors.store import VectorStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.qdrant,
]


def _unit_vector(seed: int, dimension: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(dimension).astype(np.float32)
    return vector / np.linalg.norm(vector)


@pytest.fixture
def qdrant_store() -> VectorStore:
    url = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
    try:
        response = httpx.get(url, timeout=2)
        response.raise_for_status()
    except Exception as exc:
        pytest.skip(f"Qdrant is not reachable at {url}: {exc}")

    store = VectorStore(url=url)
    if not store.available:
        pytest.skip(store.last_error or "Qdrant is unavailable")
    yield store
    store.close()


def test_context_upsert_search_and_delete_roundtrip(qdrant_store: VectorStore) -> None:
    image_id = f"pytest-{uuid.uuid4()}"
    query = _unit_vector(1, CLIP_VECTOR_DIM)
    other = _unit_vector(2, CLIP_VECTOR_DIM)
    other_id = f"pytest-{uuid.uuid4()}"
    try:
        qdrant_store.upsert_context(image_id, query, model_version="pytest")
        qdrant_store.upsert_context(other_id, other, model_version="pytest")
        hits = qdrant_store.search_context(query, [image_id, other_id], k=2)
        assert hits
        assert hits[0].image_id == image_id
        assert hits[0].score == pytest.approx(1.0, abs=1e-5)

        qdrant_store.delete_for_image(image_id)
        remaining = qdrant_store.search_context(query, [image_id], k=1)
        assert remaining == []
    finally:
        qdrant_store.delete_for_image(image_id)
        qdrant_store.delete_for_image(other_id)


def test_faces_upsert_search_scroll_and_delete_roundtrip(
    qdrant_store: VectorStore,
) -> None:
    image_id = f"pytest-{uuid.uuid4()}"
    face_id = f"pytest-{uuid.uuid4()}"
    vector = _unit_vector(3, FACE_VECTOR_DIM)
    try:
        qdrant_store.upsert_faces(
            [(face_id, image_id, vector)],
            model_version="pytest",
        )
        hits = qdrant_store.search_faces(
            vector,
            [image_id],
            limit=5,
            score_threshold=0.99,
        )
        assert [(hit.face_id, hit.image_id) for hit in hits] == [(face_id, image_id)]

        stored = qdrant_store.scroll_face_vectors([face_id])
        assert face_id in stored
        assert np.allclose(stored[face_id], vector, atol=1e-6)

        qdrant_store.delete_face_points([face_id])
        assert qdrant_store.scroll_face_vectors([face_id]) == {}
    finally:
        qdrant_store.delete_for_image(image_id)


def test_vector_dimensions_are_validated_before_qdrant_call(
    qdrant_store: VectorStore,
) -> None:
    with pytest.raises(ValueError, match=str(CLIP_VECTOR_DIM)):
        qdrant_store.upsert_context(
            f"pytest-{uuid.uuid4()}",
            np.ones(CLIP_VECTOR_DIM - 1, dtype=np.float32),
            model_version="pytest",
        )
    with pytest.raises(ValueError, match=str(FACE_VECTOR_DIM)):
        qdrant_store.upsert_faces(
            [
                (
                    f"pytest-{uuid.uuid4()}",
                    f"pytest-{uuid.uuid4()}",
                    np.ones(FACE_VECTOR_DIM - 1, dtype=np.float32),
                )
            ],
            model_version="pytest",
        )


def test_unavailable_qdrant_fails_fast_without_hanging() -> None:
    store = VectorStore(url="http://127.0.0.1:1")
    try:
        assert not store.available
        assert store.last_error is not None
    finally:
        store.close()
