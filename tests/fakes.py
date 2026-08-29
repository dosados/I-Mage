from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ml.embeddings.base import EmbeddingModel
from ml.faces.base import Face, FaceRecognizer
from ml.objects.base import Detection, ObjectsRetriever
from vectors.types import ContextHit, FaceHit


def unit_vec(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(dim).astype(np.float32)
    return vector / (np.linalg.norm(vector) + 1e-8)


class FakeClip(EmbeddingModel):
    def __init__(self, *, dim: int = 8, name: str = "fake-clip") -> None:
        self.model_name = name
        self.dim = dim
        self.encode_image_calls: list[Path] = []
        self.encode_images_calls: list[list[Path]] = []
        self.encode_text_calls: list[str] = []

    def encode_image(self, image) -> np.ndarray:
        path = Path(image)
        self.encode_image_calls.append(path)
        return unit_vec(hash(path.name) % 10_000, self.dim)

    def encode_text(self, text: str) -> np.ndarray:
        self.encode_text_calls.append(text)
        return unit_vec(hash(text) % 10_000, self.dim)

    def encode_images(self, images) -> list[np.ndarray]:
        paths = [Path(image) for image in images]
        self.encode_images_calls.append(paths)
        return [self.encode_image(path) for path in paths]


class FakeYolo(ObjectsRetriever):
    def __init__(self, *, name: str = "fake-yolo") -> None:
        self.model_name = name
        self.detect_calls: list[Path] = []
        self.detect_batch_calls: list[list[Path]] = []
        self._by_name: dict[str, list[Detection]] = {}

    def set_detections(self, path: Path, detections: list[Detection]) -> None:
        self._by_name[Path(path).name] = detections

    def detect(self, image) -> list[Detection]:
        path = Path(image)
        self.detect_calls.append(path)
        return list(self._by_name.get(path.name, [
            Detection(label="person", confidence=0.9, bbox=(0, 0, 10, 10)),
        ]))

    def detect_labels(self, image) -> list[str]:
        return sorted({d.label for d in self.detect(image)})

    def detect_batch(self, images) -> list[list[Detection]]:
        paths = [Path(image) for image in images]
        self.detect_batch_calls.append(paths)
        return [self.detect(path) for path in paths]


class FakeFaces(FaceRecognizer):
    def __init__(self, *, dim: int = 8, name: str = "fake-arcface") -> None:
        self.model_name = name
        self._dim = dim
        self.analyze_calls: list = []
        self.analyze_batch_calls: list[list] = []
        self._by_key: dict[str, list[Face]] = {}

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def set_faces(self, key: str, faces: list[Face]) -> None:
        self._by_key[key] = faces

    def analyze(self, image) -> list[Face]:
        self.analyze_calls.append(image)
        if isinstance(image, (str, Path)):
            key = Path(image).name
        else:
            key = "upload"
        if key in self._by_key:
            faces = list(self._by_key[key])
        else:
            faces = [
                Face(
                    bbox=(0, 0, 10, 10),
                    detection_score=0.99,
                    embedding=unit_vec(hash(str(key)) % 10_000, self._dim),
                )
            ]
        faces.sort(key=lambda item: item.detection_score, reverse=True)
        return faces

    def analyze_batch(self, images, *, should_stop=None) -> list[list[Face]]:
        paths = list(images)
        self.analyze_batch_calls.append(paths)
        results: list[list[Face]] = []
        for image in paths:
            if should_stop is not None and should_stop():
                raise InterruptedError("face analysis stopped")
            results.append(self.analyze(image))
        return results


class FakeVectorStore:
    """In-memory stand-in for ``vectors.store.VectorStore`` used in unit tests."""

    def __init__(self, *, available: bool = True) -> None:
        self._available = available
        self.last_error: str | None = None if available else "offline"
        self.context: dict[str, np.ndarray] = {}
        self.faces: dict[str, tuple[str, np.ndarray]] = {}  # face_id -> (image_id, vec)
        self.upsert_context_calls = 0
        self.upsert_faces_calls = 0
        self.search_context_calls = 0
        self.search_faces_calls = 0

    @property
    def available(self) -> bool:
        return self._available

    def close(self) -> None:
        return None

    def upsert_context(
        self, image_id: str, embedding: np.ndarray, *, model_version: str
    ) -> None:
        self.upsert_context_calls += 1
        self.context[image_id] = np.asarray(embedding, dtype=np.float32)

    def upsert_contexts(
        self,
        items: Sequence[tuple[str, np.ndarray]],
        *,
        model_version: str,
    ) -> None:
        self.upsert_context_calls += 1
        for image_id, embedding in items:
            self.context[image_id] = np.asarray(embedding, dtype=np.float32)

    def upsert_faces(
        self,
        faces: Sequence[tuple[str, str, np.ndarray]],
        *,
        model_version: str,
    ) -> None:
        self.upsert_faces_calls += 1
        for face_id, image_id, embedding in faces:
            self.faces[face_id] = (image_id, np.asarray(embedding, dtype=np.float32))

    def delete_for_image(self, image_id: str) -> None:
        self.context.pop(image_id, None)
        self.faces = {
            fid: (iid, vec)
            for fid, (iid, vec) in self.faces.items()
            if iid != image_id
        }

    def delete_face_points(self, face_ids: set[str] | list[str]) -> None:
        for face_id in face_ids:
            self.faces.pop(face_id, None)

    def search_context(
        self,
        query_embedding: np.ndarray,
        image_ids: Sequence[str] | None,
        *,
        k: int,
    ) -> list[ContextHit]:
        self.search_context_calls += 1
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        allowed = set(image_ids) if image_ids is not None else None
        scored: list[ContextHit] = []
        for image_id, vector in self.context.items():
            if allowed is not None and image_id not in allowed:
                continue
            score = float(vector @ query / (np.linalg.norm(vector) * np.linalg.norm(query) + 1e-8))
            scored.append(ContextHit(image_id=image_id, score=score))
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:k]

    def search_faces(
        self,
        query_embedding: np.ndarray,
        image_ids: Sequence[str] | None,
        *,
        limit: int,
        score_threshold: float | None = None,
    ) -> list[FaceHit]:
        self.search_faces_calls += 1
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        allowed = set(image_ids) if image_ids is not None else None
        scored: list[FaceHit] = []
        for face_id, (image_id, vector) in self.faces.items():
            if allowed is not None and image_id not in allowed:
                continue
            score = float(vector @ query / (np.linalg.norm(vector) * np.linalg.norm(query) + 1e-8))
            if score_threshold is not None and score < score_threshold:
                continue
            scored.append(FaceHit(face_id=face_id, image_id=image_id, score=score))
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:limit]
