import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from PIL import Image

import numpy as np

from ml.faces.base import FaceRecognizer

ImageInput = str | Path | Image.Image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryFaceEmbedding:
    embedding: np.ndarray
    detection_score: float


@dataclass(frozen=True)
class FaceMatch:
    path: Path
    score: float


@dataclass(frozen=True)
class FaceSearchResult:
    matches: list[FaceMatch]


def encode_query_face(
    image: ImageInput,
    recognizer: FaceRecognizer,
) -> QueryFaceEmbedding:
    faces = recognizer.analyze(image)
    if not faces:
        raise ValueError("no face found in query image")

    face = faces[0]
    return QueryFaceEmbedding(
        embedding=face.embedding,
        detection_score=face.detection_score,
    )


def _normalize_query_embedding(
    query_embedding: np.ndarray,
    *,
    expected_dim: int,
) -> np.ndarray:
    vector = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    if vector.shape[0] != expected_dim:
        raise ValueError(
            f"query embedding must have dimension {expected_dim}, got {vector.shape[0]}"
        )

    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise ValueError("query embedding must not be a zero vector")

    return (vector / norm).astype(np.float32)


def search_by_face(
    query_embedding: np.ndarray,
    paths: Iterable[str | Path],
    recognizer: FaceRecognizer,
    *,
    k: int = 10,
    threshold: float = 0.4,
) -> FaceSearchResult:
    image_paths = [Path(path) for path in paths]
    if not image_paths:
        raise ValueError("no image paths provided")

    query = _normalize_query_embedding(
        query_embedding,
        expected_dim=recognizer.embedding_dim,
    )

    matches: list[FaceMatch] = []

    for image_path in image_paths:
        try:
            faces = recognizer.analyze(image_path)
        except Exception:
            logger.exception("failed to analyze faces, skipping: %s", image_path)
            continue

        best_score: float | None = None
        for face in faces:
            score = float(face.embedding @ query)
            if score >= threshold and (best_score is None or score > best_score):
                best_score = score

        if best_score is not None:
            matches.append(FaceMatch(path=image_path, score=best_score))

    matches.sort(key=lambda item: item.score, reverse=True)
    if k is not None:
        matches = matches[:k]

    if not matches:
        raise ValueError("no matching faces found")

    return FaceSearchResult(matches=matches)
