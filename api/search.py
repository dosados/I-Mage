from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from api.paths import resolve_scoped_image_paths
from db.database import Database
from db.hash import resolved_path_key
from db.types import MODULE_CLIP, MODULE_FACES, MODULE_YOLO
from indexing.clip import index_clip_gap
from indexing.faces import index_faces_gap
from indexing.ml_lock import get_ml_lock
from indexing.yolo import index_yolo_gap
from ml.embeddings.base import EmbeddingModel
from ml.embeddings.service import ImageMatch, SearchResult, search_by_description
from ml.faces.base import FaceRecognizer
from ml.faces.service import (
    FaceMatch,
    FaceSearchResult,
    QueryFaceEmbedding,
    encode_query_face,
    search_by_face,
)
from ml.objects.base import ImageInput, ObjectsRetriever
from vectors.store import VectorStore

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, str, dict[str, Any]], None]


@dataclass(frozen=True)
class ClassSearchMatch:
    path: Path
    confidence: float


@dataclass(frozen=True)
class ClassSearchResult:
    label: str
    matches: list[ClassSearchMatch]


def _resolve_paths(db: Database, *, limit: int | None = None) -> list[Path]:
    config = db.get_scan_config()
    return resolve_scoped_image_paths(
        config.include_directories,
        config.ignore_globs,
        limit=limit,
    )


def _emit(
    on_progress: ProgressCallback | None,
    stage: str,
    status: str,
    **payload: Any,
) -> None:
    if on_progress is not None:
        on_progress(stage, status, payload)


def _gap_paths(db: Database, paths: list[Path], module: str) -> list[Path]:
    """Paths in scope whose module status isn't DONE — one flat query, no ORM graph."""
    done = db.images.done_paths_for_module(module, paths)
    return [path for path in paths if resolved_path_key(path) not in done]


def run_search_by_description(
    query: str,
    model: EmbeddingModel,
    db: Database,
    vector_store: VectorStore,
    *,
    limit: int | None = None,
    k: int = 1,
    on_progress: ProgressCallback | None = None,
) -> SearchResult:
    _emit(on_progress, "reconcile", "running")
    paths = _resolve_paths(db, limit=limit)
    path_set = set(paths)
    model_version = model.model_name

    with db:
        db.reconcile_paths(path_set, remove_missing=False)
        gap_paths = _gap_paths(db, paths, MODULE_CLIP)
    _emit(on_progress, "reconcile", "done", total=len(paths), gap=len(gap_paths))

    if gap_paths and vector_store.available:
        _emit(on_progress, "index_gap", "running", gap=len(gap_paths))
        with get_ml_lock():
            with db:
                index_clip_gap(
                    db,
                    vector_store,
                    gap_paths,
                    model,
                    model_version=model_version,
                )
        _emit(on_progress, "index_gap", "done", gap=len(gap_paths))
    else:
        _emit(on_progress, "index_gap", "done", gap=0)

    _emit(on_progress, "search", "running")
    if vector_store.available:
        query_embedding = model.encode_text(query)
        with db:
            id_to_path = db.images.id_to_path_for_scope(paths)
        hits = vector_store.search_context(
            query_embedding,
            list(id_to_path.keys()) if id_to_path else None,
            k=k,
        )
        matches = [
            ImageMatch(path=id_to_path[hit.image_id], score=hit.score)
            for hit in hits
            if hit.image_id in id_to_path
        ]
        _emit(on_progress, "search", "done", matches=len(matches))
        if matches:
            return SearchResult(query=query, matches=matches)
        # Index exists but nothing matched — do NOT fall back to embedding every file.
        return SearchResult(query=query, matches=[])

    # No Qdrant: legacy on-the-fly scan.
    result = search_by_description(query, paths, model, k=k)
    _emit(on_progress, "search", "done", matches=len(result.matches))
    return result


def run_search_by_class(
    label: str,
    retriever: ObjectsRetriever,
    db: Database,
    *,
    limit: int | None = None,
    k: int | None = None,
    on_progress: ProgressCallback | None = None,
) -> ClassSearchResult:
    normalized_label = label.strip().lower()
    if not normalized_label:
        raise ValueError("label must not be empty")

    _emit(on_progress, "reconcile", "running")
    paths = _resolve_paths(db, limit=limit)
    path_set = set(paths)
    model_version = retriever.model_name

    with db:
        db.reconcile_paths(path_set, remove_missing=False)
        gap_paths = _gap_paths(db, paths, MODULE_YOLO)
    _emit(on_progress, "reconcile", "done", total=len(paths), gap=len(gap_paths))

    if gap_paths:
        _emit(on_progress, "index_gap", "running", gap=len(gap_paths))
        with get_ml_lock():
            with db:
                index_yolo_gap(db, gap_paths, retriever, model_version=model_version)
        _emit(on_progress, "index_gap", "done", gap=len(gap_paths))
    else:
        _emit(on_progress, "index_gap", "done", gap=0)

    _emit(on_progress, "search", "running")
    with db:
        matches = db.detections.search_by_label(
            label,
            k=k,
            paths=path_set,
        )
    _emit(on_progress, "search", "done", matches=len(matches))

    return ClassSearchResult(
        label=label,
        matches=[
            ClassSearchMatch(path=match.path, confidence=match.confidence)
            for match in matches
        ],
    )


def run_embed_query_face(
    image: ImageInput,
    recognizer: FaceRecognizer,
) -> QueryFaceEmbedding:
    return encode_query_face(image, recognizer)


def _aggregate_face_hits(
    hits,
    *,
    id_to_path: dict[str, Path],
    k: int,
) -> list[FaceMatch]:
    best_by_image: dict[str, float] = {}
    for hit in hits:
        if hit.image_id not in id_to_path:
            continue
        best_by_image[hit.image_id] = max(best_by_image.get(hit.image_id, hit.score), hit.score)

    matches = [
        FaceMatch(path=id_to_path[image_id], score=score)
        for image_id, score in best_by_image.items()
    ]
    matches.sort(key=lambda item: item.score, reverse=True)
    return matches[:k]


def run_search_by_face(
    query_embedding: np.ndarray | list[float],
    recognizer: FaceRecognizer,
    db: Database,
    vector_store: VectorStore,
    *,
    limit: int | None = None,
    k: int = 10,
    threshold: float = 0.4,
    on_progress: ProgressCallback | None = None,
    allow_bruteforce_fallback: bool = False,
) -> FaceSearchResult:
    """Search faces against the Qdrant index after a cheap catalog reconcile.

    After a full Faces run the gap is empty, so this path is: reconcile → vector
    search. It must NOT re-run ArcFace over the whole catalog when Qdrant is up.
    """
    _emit(on_progress, "reconcile", "running")
    paths = _resolve_paths(db, limit=limit)
    path_set = set(paths)
    model_version = recognizer.model_name
    embedding = (
        np.asarray(query_embedding, dtype=np.float32)
        if not isinstance(query_embedding, np.ndarray)
        else query_embedding
    )

    with db:
        db.reconcile_paths(path_set, remove_missing=False)
        gap_paths = _gap_paths(db, paths, MODULE_FACES)
    _emit(on_progress, "reconcile", "done", total=len(paths), gap=len(gap_paths))

    if gap_paths and vector_store.available:
        _emit(on_progress, "index_gap", "running", gap=len(gap_paths))
        logger.info("face search indexing gap=%d before query", len(gap_paths))
        with get_ml_lock():
            with db:
                index_faces_gap(
                    db,
                    vector_store,
                    gap_paths,
                    recognizer,
                    model_version=model_version,
                )
        _emit(on_progress, "index_gap", "done", gap=len(gap_paths))
    else:
        _emit(on_progress, "index_gap", "done", gap=0)

    _emit(on_progress, "search", "running")
    if vector_store.available:
        with db:
            id_to_path = db.images.id_to_path_for_scope(paths)
        if not id_to_path:
            _emit(on_progress, "search", "done", matches=0)
            raise ValueError("no indexed images in scope — run Faces indexing first")

        face_limit = max(k * 20, 100)
        hits = vector_store.search_faces(
            embedding,
            list(id_to_path.keys()),
            limit=face_limit,
            score_threshold=threshold,
        )
        matches = _aggregate_face_hits(hits, id_to_path=id_to_path, k=k)
        _emit(on_progress, "search", "done", matches=len(matches))
        # Empty is a valid result (nothing above threshold) — not a cue to
        # re-scan every photo with ArcFace.
        return FaceSearchResult(matches=matches)

    if not allow_bruteforce_fallback:
        _emit(on_progress, "search", "done", matches=0)
        raise ValueError("Qdrant unavailable — face index required for search")

    result = search_by_face(
        embedding,
        paths,
        recognizer,
        k=k,
        threshold=threshold,
    )
    _emit(on_progress, "search", "done", matches=len(result.matches))
    return result


def iter_description_search_events(
    query: str,
    model: EmbeddingModel,
    db: Database,
    vector_store: VectorStore,
    *,
    limit: int | None = None,
    k: int = 1,
) -> Iterator[dict[str, Any]]:
    try:
        yield {"stage": "reconcile", "status": "running"}
        paths = _resolve_paths(db, limit=limit)
        path_set = set(paths)
        model_version = model.model_name
        with db:
            db.reconcile_paths(path_set, remove_missing=False)
            gap_paths = _gap_paths(db, paths, MODULE_CLIP)
        yield {
            "stage": "reconcile",
            "status": "done",
            "total": len(paths),
            "gap": len(gap_paths),
        }

        if gap_paths and vector_store.available:
            yield {"stage": "index_gap", "status": "running", "gap": len(gap_paths)}
            with get_ml_lock():
                with db:
                    index_clip_gap(
                        db,
                        vector_store,
                        gap_paths,
                        model,
                        model_version=model_version,
                    )
            yield {"stage": "index_gap", "status": "done", "gap": len(gap_paths)}
        else:
            yield {"stage": "index_gap", "status": "done", "gap": 0}

        yield {"stage": "search", "status": "running"}
        if vector_store.available:
            query_embedding = model.encode_text(query)
            with db:
                id_to_path = db.images.id_to_path_for_scope(paths)
            hits = vector_store.search_context(
                query_embedding,
                list(id_to_path.keys()) if id_to_path else None,
                k=k,
            )
            matches = [
                {"path": str(id_to_path[hit.image_id]), "score": hit.score}
                for hit in hits
                if hit.image_id in id_to_path
            ]
        else:
            result = search_by_description(query, paths, model, k=k)
            matches = [
                {"path": str(match.path), "score": match.score} for match in result.matches
            ]
        yield {"stage": "search", "status": "done", "matches": len(matches)}
        yield {"stage": "done", "status": "done", "matches": matches, "query": query}
    except ValueError as exc:
        yield {"stage": "error", "status": "failed", "detail": str(exc)}


def iter_class_search_events(
    label: str,
    retriever: ObjectsRetriever,
    db: Database,
    *,
    limit: int | None = None,
    k: int | None = None,
) -> Iterator[dict[str, Any]]:
    try:
        normalized = label.strip().lower()
        if not normalized:
            raise ValueError("label must not be empty")

        yield {"stage": "reconcile", "status": "running"}
        paths = _resolve_paths(db, limit=limit)
        path_set = set(paths)
        model_version = retriever.model_name
        with db:
            db.reconcile_paths(path_set, remove_missing=False)
            gap_paths = _gap_paths(db, paths, MODULE_YOLO)
        yield {
            "stage": "reconcile",
            "status": "done",
            "total": len(paths),
            "gap": len(gap_paths),
        }

        if gap_paths:
            yield {"stage": "index_gap", "status": "running", "gap": len(gap_paths)}
            with get_ml_lock():
                with db:
                    index_yolo_gap(
                        db, gap_paths, retriever, model_version=model_version
                    )
            yield {"stage": "index_gap", "status": "done", "gap": len(gap_paths)}
        else:
            yield {"stage": "index_gap", "status": "done", "gap": 0}

        yield {"stage": "search", "status": "running"}
        with db:
            found = db.detections.search_by_label(label, k=k, paths=path_set)
        matches = [
            {"path": str(match.path), "confidence": match.confidence} for match in found
        ]
        yield {"stage": "search", "status": "done", "matches": len(matches)}
        yield {
            "stage": "done",
            "status": "done",
            "matches": matches,
            "label": label,
        }
    except ValueError as exc:
        yield {"stage": "error", "status": "failed", "detail": str(exc)}


def iter_face_search_events(
    query_embedding: np.ndarray | list[float],
    recognizer: FaceRecognizer,
    db: Database,
    vector_store: VectorStore,
    *,
    limit: int | None = None,
    k: int = 10,
    threshold: float = 0.4,
) -> Iterator[dict[str, Any]]:
    """Yield NDJSON progress after each stage so the UI can update live."""
    try:
        yield {"stage": "reconcile", "status": "running"}
        paths = _resolve_paths(db, limit=limit)
        path_set = set(paths)
        model_version = recognizer.model_name
        embedding = (
            np.asarray(query_embedding, dtype=np.float32)
            if not isinstance(query_embedding, np.ndarray)
            else query_embedding
        )

        with db:
            db.reconcile_paths(path_set, remove_missing=False)
            gap_paths = _gap_paths(db, paths, MODULE_FACES)
        yield {
            "stage": "reconcile",
            "status": "done",
            "total": len(paths),
            "gap": len(gap_paths),
        }

        if gap_paths and vector_store.available:
            yield {"stage": "index_gap", "status": "running", "gap": len(gap_paths)}
            logger.info("face search indexing gap=%d before query", len(gap_paths))
            with get_ml_lock():
                with db:
                    index_faces_gap(
                        db,
                        vector_store,
                        gap_paths,
                        recognizer,
                        model_version=model_version,
                    )
            yield {"stage": "index_gap", "status": "done", "gap": len(gap_paths)}
        else:
            yield {"stage": "index_gap", "status": "done", "gap": 0}

        yield {"stage": "search", "status": "running"}
        if not vector_store.available:
            raise ValueError("Qdrant unavailable — face index required for search")

        with db:
            id_to_path = db.images.id_to_path_for_scope(paths)
        if not id_to_path:
            raise ValueError("no indexed images in scope — run Faces indexing first")

        hits = vector_store.search_faces(
            embedding,
            list(id_to_path.keys()),
            limit=max(k * 20, 100),
            score_threshold=threshold,
        )
        matches = _aggregate_face_hits(hits, id_to_path=id_to_path, k=k)
        yield {"stage": "search", "status": "done", "matches": len(matches)}
        yield {
            "stage": "done",
            "status": "done",
            "matches": [
                {"path": str(match.path), "score": match.score} for match in matches
            ],
        }
    except ValueError as exc:
        yield {"stage": "error", "status": "failed", "detail": str(exc)}
