from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from vectors.config import (
    CLIP_VECTOR_DIM,
    CONTEXT_COLLECTION,
    FACE_VECTOR_DIM,
    FACES_COLLECTION,
    resolve_qdrant_url,
)
from vectors.ids import context_point_id, face_point_id
from vectors.types import ContextHit, FaceHit

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, url: str | None = None) -> None:
        self.url = resolve_qdrant_url(url)
        self._client: QdrantClient | None = None
        self._available = False
        self.last_error: str | None = None
        self._connect()

    @property
    def available(self) -> bool:
        return self._available and self._client is not None

    def _connect(self) -> None:
        try:
            client = QdrantClient(url=self.url)
            client.get_collections()
            self._client = client
            self._available = True
            self.last_error = None
            self.ensure_collections()
        except Exception as exc:
            self.last_error = (
                f"Qdrant-сервер недоступен ({self.url}): {exc}. "
                "Запустите его: docker compose up -d (см. docker-compose.yml)."
            )
            logger.exception("qdrant server unavailable at %s", self.url)
            self._client = None
            self._available = False

    def ensure_collections(self) -> None:
        if not self.available:
            return

        assert self._client is not None
        existing = {collection.name for collection in self._client.get_collections().collections}

        if CONTEXT_COLLECTION not in existing:
            self._client.create_collection(
                collection_name=CONTEXT_COLLECTION,
                vectors_config=VectorParams(size=CLIP_VECTOR_DIM, distance=Distance.COSINE),
            )

        if FACES_COLLECTION not in existing:
            self._client.create_collection(
                collection_name=FACES_COLLECTION,
                vectors_config=VectorParams(size=FACE_VECTOR_DIM, distance=Distance.COSINE),
            )

        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        """Index image_id / face_id so filtered delete/search doesn't full-scan.

        Filtered deletes (replace faces for an image) rely on these to avoid a
        linear scan of the whole collection.
        """
        assert self._client is not None
        targets = {
            CONTEXT_COLLECTION: ("image_id",),
            FACES_COLLECTION: ("image_id", "face_id"),
        }
        for collection, fields in targets.items():
            for field in fields:
                try:
                    self._client.create_payload_index(
                        collection_name=collection,
                        field_name=field,
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    logger.debug("payload index %s.%s already exists or failed", collection, field)

    def delete_for_image(self, image_id: str) -> None:
        if not self.available:
            return

        assert self._client is not None
        image_filter = Filter(
            must=[FieldCondition(key="image_id", match=MatchAny(any=[image_id]))]
        )

        for collection in (CONTEXT_COLLECTION, FACES_COLLECTION):
            self._client.delete(
                collection_name=collection,
                points_selector=FilterSelector(filter=image_filter),
            )

    def upsert_context(
        self,
        image_id: str,
        embedding: np.ndarray,
        *,
        model_version: str,
    ) -> None:
        if not self.available:
            raise RuntimeError("qdrant is not available")

        assert self._client is not None
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.shape[0] != CLIP_VECTOR_DIM:
            raise ValueError(f"context vector must have dimension {CLIP_VECTOR_DIM}")

        self._client.upsert(
            collection_name=CONTEXT_COLLECTION,
            points=[
                PointStruct(
                    id=context_point_id(image_id, model_version),
                    vector=vector.tolist(),
                    payload={
                        "image_id": image_id,
                        "model_version": model_version,
                    },
                )
            ],
        )

    def upsert_faces(
        self,
        faces: Sequence[tuple[str, str, np.ndarray]],
        *,
        model_version: str,
    ) -> None:
        if not self.available:
            raise RuntimeError("qdrant is not available")
        if not faces:
            return

        assert self._client is not None
        points: list[PointStruct] = []
        for face_id, image_id, embedding in faces:
            vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
            if vector.shape[0] != FACE_VECTOR_DIM:
                raise ValueError(f"face vector must have dimension {FACE_VECTOR_DIM}")

            points.append(
                PointStruct(
                    id=face_point_id(face_id),
                    vector=vector.tolist(),
                    payload={
                        "face_id": face_id,
                        "image_id": image_id,
                        "model_version": model_version,
                    },
                )
            )

        self._client.upsert(collection_name=FACES_COLLECTION, points=points)

    def delete_face_points(self, face_ids: Sequence[str]) -> None:
        """Delete specific face points by id (cheap: no filter scan).

        Used to drop only the faces that disappeared on re-detection. Deleting by
        explicit point id avoids the per-image filtered delete that otherwise
        dominates re-index throughput.
        """
        if not self.available or not face_ids:
            return
        assert self._client is not None
        self._client.delete(
            collection_name=FACES_COLLECTION,
            points_selector=[face_point_id(face_id) for face_id in face_ids],
        )

    def search_context(
        self,
        query_embedding: np.ndarray,
        image_ids: Sequence[str] | None,
        *,
        k: int,
    ) -> list[ContextHit]:
        if not self.available:
            return []

        assert self._client is not None
        vector = np.asarray(query_embedding, dtype=np.float32).reshape(-1).tolist()
        # MatchAny with tens of thousands of UUIDs is slow/fragile. For large
        # scopes search globally and filter in Python instead.
        query_filter = None
        if image_ids is not None and 0 < len(image_ids) <= 1000:
            query_filter = Filter(
                must=[FieldCondition(key="image_id", match=MatchAny(any=list(image_ids)))]
            )
        response = self._client.query_points(
            collection_name=CONTEXT_COLLECTION,
            query=vector,
            query_filter=query_filter,
            limit=k if query_filter is not None else max(k * 20, 50),
            with_payload=True,
        )

        allowed = set(image_ids) if image_ids is not None else None
        hits: list[ContextHit] = []
        for point in response.points:
            payload = point.payload or {}
            image_id = payload.get("image_id")
            if not isinstance(image_id, str):
                continue
            if allowed is not None and image_id not in allowed:
                continue
            hits.append(ContextHit(image_id=image_id, score=float(point.score)))
            if len(hits) >= k:
                break

        return hits

    def search_faces(
        self,
        query_embedding: np.ndarray,
        image_ids: Sequence[str] | None,
        *,
        limit: int,
        score_threshold: float | None = None,
    ) -> list[FaceHit]:
        if not self.available:
            return []

        assert self._client is not None
        vector = np.asarray(query_embedding, dtype=np.float32).reshape(-1).tolist()
        query_filter = None
        fetch_limit = limit
        if image_ids is not None and 0 < len(image_ids) <= 1000:
            query_filter = Filter(
                must=[FieldCondition(key="image_id", match=MatchAny(any=list(image_ids)))]
            )
        else:
            # Over-fetch so post-filtering by scope still yields ``limit`` hits.
            fetch_limit = max(limit * 5, 200)

        response = self._client.query_points(
            collection_name=FACES_COLLECTION,
            query=vector,
            query_filter=query_filter,
            limit=fetch_limit,
            score_threshold=score_threshold,
            with_payload=True,
        )

        allowed = set(image_ids) if image_ids is not None else None
        hits: list[FaceHit] = []
        for point in response.points:
            payload = point.payload or {}
            face_id = payload.get("face_id")
            image_id = payload.get("image_id")
            if not isinstance(face_id, str) or not isinstance(image_id, str):
                continue
            if allowed is not None and image_id not in allowed:
                continue
            hits.append(
                FaceHit(
                    face_id=face_id,
                    image_id=image_id,
                    score=float(point.score),
                )
            )
            if len(hits) >= limit:
                break

        return hits

    def scroll_face_vectors(self, face_ids: Sequence[str]) -> dict[str, np.ndarray]:
        if not self.available or not face_ids:
            return {}

        assert self._client is not None
        vectors: dict[str, np.ndarray] = {}
        batch_size = 256
        for offset in range(0, len(face_ids), batch_size):
            batch = face_ids[offset : offset + batch_size]
            point_ids = [face_point_id(face_id) for face_id in batch]
            points = self._client.retrieve(
                collection_name=FACES_COLLECTION,
                ids=point_ids,
                with_vectors=True,
            )
            for point in points:
                payload = point.payload or {}
                face_id = payload.get("face_id")
                if not isinstance(face_id, str) or point.vector is None:
                    continue
                vectors[face_id] = np.asarray(point.vector, dtype=np.float32)

        return vectors

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        self._available = False


def create_vector_store(url: str | None = None) -> VectorStore:
    return VectorStore(url=url)
