from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from api.keywords import extract_labels_from_query, merge_labels
from indexing.gpu_scheduler import GpuScheduler, get_gpu_scheduler
from ml.embeddings.base import EmbeddingModel
from ml.embeddings.service import ImageMatch, SearchResult
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
    with db:
        return [record.path for record in db.images.list_all(limit=limit)]


def _emit(
    on_progress: ProgressCallback | None,
    stage: str,
    status: str,
    **payload: Any,
) -> None:
    if on_progress is not None:
        on_progress(stage, status, payload)


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
    paths = _resolve_paths(db, limit=limit)
    _emit(on_progress, "catalog", "done", total=len(paths))

    _emit(on_progress, "search", "running")
    if not vector_store.available:
        raise ValueError("Qdrant unavailable — CLIP index required for search")
    with get_gpu_scheduler().acquire(
        "search:clip-text",
        priority=GpuScheduler.INTERACTIVE,
    ):
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
    return SearchResult(query=query, matches=matches)


def run_search_by_class(
    label: str,
    retriever: ObjectsRetriever,
    db: Database,
    *,
    limit: int | None = None,
    k: int | None = None,
    on_progress: ProgressCallback | None = None,
) -> ClassSearchResult:
    del retriever
    normalized_label = label.strip().lower()
    if not normalized_label:
        raise ValueError("label must not be empty")

    paths = _resolve_paths(db, limit=limit)
    path_set = set(paths) if limit is not None else None
    _emit(on_progress, "catalog", "done", total=len(paths))

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
    with get_gpu_scheduler().acquire(
        "search:face-query",
        priority=GpuScheduler.INTERACTIVE,
    ):
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
    """Search only the existing face index; catalog indexing is a separate job."""
    paths = _resolve_paths(db, limit=limit)
    _emit(on_progress, "catalog", "done", total=len(paths))
    embedding = (
        np.asarray(query_embedding, dtype=np.float32)
        if not isinstance(query_embedding, np.ndarray)
        else query_embedding
    )

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


@dataclass(frozen=True)
class UnifiedSearchMatch:
    path: Path
    clip_score: float | None
    yolo: dict[str, float]
    sources: list[str]
    rank_score: float


@dataclass(frozen=True)
class UnifiedSearchResult:
    query: str
    labels: list[str]
    matches: list[UnifiedSearchMatch]


def _resolve_unified_labels(query: str, labels: list[str] | None) -> list[str]:
    auto = extract_labels_from_query(query) if labels is None else []
    explicit = labels or []
    return merge_labels(auto, explicit)


def _rank_unified(
    *,
    clip_score: float | None,
    yolo: dict[str, float],
    sources: list[str],
) -> float:
    """Return an ordering score for the unified search feed.

    Every selected object label matched by an image is a separate relevance
    tier. A semantic score can promote an image within its object-match tier,
    but cannot move an image matching fewer selected labels above it.
    """
    del sources
    object_matches = len(yolo)
    clip = clip_score or 0.0
    yolo_max = max(yolo.values()) if yolo else 0.0
    return object_matches * 100.0 + clip * 10.0 + yolo_max


def _merge_unified_matches(
    clip_matches: list[ImageMatch],
    yolo_matches_by_label: dict[str, list[ClassSearchMatch]],
    *,
    k: int,
) -> list[UnifiedSearchMatch]:
    by_path: dict[str, UnifiedSearchMatch] = {}

    for match in clip_matches:
        key = str(match.path)
        sources = ["semantic"]
        yolo: dict[str, float] = {}
        entry = UnifiedSearchMatch(
            path=match.path,
            clip_score=match.score,
            yolo=yolo,
            sources=sources,
            rank_score=_rank_unified(
                clip_score=match.score, yolo=yolo, sources=sources
            ),
        )
        by_path[key] = entry

    for label, matches in yolo_matches_by_label.items():
        for match in matches:
            key = str(match.path)
            existing = by_path.get(key)
            if existing is None:
                yolo = {label: match.confidence}
                sources = ["object"]
                by_path[key] = UnifiedSearchMatch(
                    path=match.path,
                    clip_score=None,
                    yolo=yolo,
                    sources=sources,
                    rank_score=_rank_unified(
                        clip_score=None, yolo=yolo, sources=sources
                    ),
                )
                continue

            yolo = dict(existing.yolo)
            yolo[label] = max(yolo.get(label, match.confidence), match.confidence)
            sources = list(existing.sources)
            if "object" not in sources:
                sources.append("object")
            by_path[key] = UnifiedSearchMatch(
                path=existing.path,
                clip_score=existing.clip_score,
                yolo=yolo,
                sources=sources,
                rank_score=_rank_unified(
                    clip_score=existing.clip_score, yolo=yolo, sources=sources
                ),
            )

    merged = sorted(by_path.values(), key=lambda item: item.rank_score, reverse=True)
    return merged[:k]


def run_unified_search(
    query: str,
    model: EmbeddingModel,
    retriever: ObjectsRetriever,
    db: Database,
    vector_store: VectorStore,
    *,
    labels: list[str] | None = None,
    limit: int | None = None,
    k: int = 10,
    on_progress: ProgressCallback | None = None,
) -> UnifiedSearchResult:
    del retriever
    normalized_query = query.strip()
    active_labels = _resolve_unified_labels(normalized_query, labels)
    if not normalized_query and not active_labels:
        raise ValueError("query or object label is required")

    paths = _resolve_paths(db, limit=limit)
    path_set = set(paths) if limit is not None else None
    _emit(on_progress, "catalog", "done", total=len(paths))

    clip_matches: list[ImageMatch] = []
    if normalized_query:
        _emit(on_progress, "search", "running", source="semantic")
        if vector_store.available and paths:
            with get_gpu_scheduler().acquire(
                "search:clip-text",
                priority=GpuScheduler.INTERACTIVE,
            ):
                query_embedding = model.encode_text(normalized_query)
            with db:
                id_to_path = db.images.id_to_path_for_scope(paths)
            hits = vector_store.search_context(
                query_embedding,
                list(id_to_path.keys()) if id_to_path else None,
                k=k,
            )
            clip_matches = [
                ImageMatch(path=id_to_path[hit.image_id], score=hit.score)
                for hit in hits
                if hit.image_id in id_to_path
            ]
        elif not vector_store.available:
            raise ValueError("Qdrant unavailable — CLIP index required for search")
        _emit(on_progress, "search", "done", source="semantic", matches=len(clip_matches))

    yolo_by_label: dict[str, list[ClassSearchMatch]] = {}
    for label in active_labels:
        _emit(on_progress, "search", "running", source="object", label=label)
        with db:
            found = db.detections.search_by_label(label, k=k, paths=path_set)
        yolo_by_label[label] = [
            ClassSearchMatch(path=match.path, confidence=match.confidence)
            for match in found
        ]
        _emit(
            on_progress,
            "search",
            "done",
            source="object",
            label=label,
            matches=len(yolo_by_label[label]),
        )

    merged = _merge_unified_matches(clip_matches, yolo_by_label, k=k)
    return UnifiedSearchResult(
        query=normalized_query,
        labels=active_labels,
        matches=merged,
    )


def _unified_matches_payload(matches: list[UnifiedSearchMatch]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(match.path),
            "clip_score": match.clip_score,
            "yolo": match.yolo,
            "sources": match.sources,
            "rank_score": match.rank_score,
        }
        for match in matches
    ]


def iter_unified_search_events(
    query: str,
    model: EmbeddingModel,
    retriever: ObjectsRetriever,
    db: Database,
    vector_store: VectorStore,
    *,
    labels: list[str] | None = None,
    limit: int | None = None,
    k: int = 10,
) -> Iterator[dict[str, Any]]:
    try:
        yield from _iter_unified_search_events_impl(
            query,
            model,
            retriever,
            db,
            vector_store,
            labels=labels,
            limit=limit,
            k=k,
        )
    except ValueError as exc:
        yield {"stage": "error", "status": "failed", "detail": str(exc)}


def _iter_unified_search_events_impl(
    query: str,
    model: EmbeddingModel,
    retriever: ObjectsRetriever,
    db: Database,
    vector_store: VectorStore,
    *,
    labels: list[str] | None = None,
    limit: int | None = None,
    k: int = 10,
) -> Iterator[dict[str, Any]]:
    yield {"stage": "search", "status": "running"}
    result = run_unified_search(
        query,
        model,
        retriever,
        db,
        vector_store,
        labels=labels,
        limit=limit,
        k=k,
    )
    yield {
        "stage": "done",
        "status": "done",
        "query": result.query,
        "labels": result.labels,
        "matches": _unified_matches_payload(result.matches),
    }


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
        yield {"stage": "search", "status": "running"}
        result = run_search_by_description(
            query,
            model,
            db,
            vector_store,
            limit=limit,
            k=k,
        )
        matches = [
            {"path": str(match.path), "score": match.score}
            for match in result.matches
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
        yield {"stage": "search", "status": "running"}
        result = run_search_by_class(
            label,
            retriever,
            db,
            limit=limit,
            k=k,
        )
        matches = [
            {"path": str(match.path), "confidence": match.confidence}
            for match in result.matches
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
    """Yield progress for an index-only face search."""
    try:
        yield {"stage": "search", "status": "running"}
        result = run_search_by_face(
            query_embedding,
            recognizer,
            db,
            vector_store,
            limit=limit,
            k=k,
            threshold=threshold,
        )
        matches = result.matches
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
