from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from api.schemas import (
    PersonDetailResponse,
    PersonFaceResponse,
    PersonListResponse,
    PersonMergeRequest,
    PersonRenameRequest,
    PersonSplitRequest,
    PersonSummaryResponse,
)
from db.types import MODULE_FACES
from indexing.executor import IndexRunConflictError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/people", tags=["people"])


def _summary_from_faces(person, faces) -> PersonSummaryResponse:
    preview_face_id = faces[0].face_id if faces else None
    preview_image_id = faces[0].image_id if faces else None
    sample = faces[:3]
    display_name = person.name if person.name else f"Человек #{person.id[:8]}"
    return PersonSummaryResponse(
        id=person.id,
        name=person.name,
        display_name=display_name,
        is_named=person.is_named,
        face_count=person.face_count,
        preview_face_id=preview_face_id,
        preview_image_id=preview_image_id,
        sample_image_ids=[face.image_id for face in sample],
        sample_image_paths=[str(face.image_path) for face in sample],
    )


def _person_summary(request: Request, person) -> PersonSummaryResponse:
    with request.app.state.db:
        faces = request.app.state.db.persons.list_faces_for_person(person.id)
    return _summary_from_faces(person, faces)


@router.get("", response_model=PersonListResponse)
def list_people(
    request: Request,
    *,
    min_face_count: int = 2,
    limit: int = 500,
    offset: int = 0,
) -> PersonListResponse:
    db = request.app.state.db
    with db:
        persons = db.persons.list_persons(
            min_face_count=min_face_count, limit=limit, offset=offset
        )
        previews = db.persons.preview_faces_for_persons([p.id for p in persons])
        total = db.persons.count_persons(min_face_count=min_face_count)
        total_all = db.persons.count_persons()
    people = [_summary_from_faces(person, previews.get(person.id, [])) for person in persons]
    return PersonListResponse(
        people=people,
        total=total,
        total_all=total_all,
        returned=len(people),
        offset=offset,
        min_face_count=min_face_count,
    )


@router.patch("/{person_id}", response_model=PersonDetailResponse)
def rename_person(person_id: str, payload: PersonRenameRequest, request: Request) -> PersonDetailResponse:
    with request.app.state.db:
        updated = request.app.state.db.persons.rename_person(person_id, payload.name)
        if updated is None:
            raise HTTPException(status_code=404, detail="person not found")
    summary = _person_summary(request, updated)
    return PersonDetailResponse(**summary.model_dump())


@router.post("/merge", response_model=PersonDetailResponse)
def merge_people(payload: PersonMergeRequest, request: Request) -> PersonDetailResponse:
    try:
        with request.app.state.db:
            merged = request.app.state.db.persons.merge_persons(
                payload.from_person_id,
                payload.to_person_id,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if merged is None:
        raise HTTPException(status_code=404, detail="person not found")
    summary = _person_summary(request, merged)
    return PersonDetailResponse(**summary.model_dump())


@router.post("/split", response_model=PersonDetailResponse)
def split_person(payload: PersonSplitRequest, request: Request) -> PersonDetailResponse:
    try:
        with request.app.state.db:
            created = request.app.state.db.persons.split_person(
                payload.person_id,
                payload.face_ids,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if created is None:
        raise HTTPException(status_code=404, detail="person not found")
    summary = _person_summary(request, created)
    return PersonDetailResponse(**summary.model_dump())


@router.get("/{person_id}/faces", response_model=list[PersonFaceResponse])
def list_person_faces(person_id: str, request: Request) -> list[PersonFaceResponse]:
    with request.app.state.db:
        person = request.app.state.db.persons.get_person(person_id)
        if person is None:
            raise HTTPException(status_code=404, detail="person not found")
        faces = request.app.state.db.persons.list_faces_for_person(person_id)
    return [
        PersonFaceResponse(
            face_id=face.face_id,
            image_id=face.image_id,
            image_path=str(face.image_path),
            bbox=list(face.bbox),
            detection_score=face.detection_score,
        )
        for face in faces
    ]


@router.get("/low-confidence")
def low_confidence_faces(
    request: Request, *, max_score: float = 0.7, limit: int = 200
) -> dict:
    db = request.app.state.db
    with db:
        total = db.faces.count_low_confidence(max_score)
        faces = db.faces.list_low_confidence(max_score, limit=limit)
    return {
        "threshold": max_score,
        "total": total,
        "returned": len(faces),
        "faces": [
            {
                "face_id": face.face_id,
                "image_id": face.image_id,
                "image_path": str(face.image_path),
                "bbox": list(face.bbox),
                "detection_score": face.detection_score,
            }
            for face in faces
        ],
    }


@router.get("/cluster/status")
def cluster_status(request: Request) -> dict:
    """Clustering state, derived from the persisted faces run (survives reload).

    Kept for backwards compatibility; the UI now reads /index/status directly.
    """
    with request.app.state.db:
        latest = request.app.state.db.index_runs.get_latest(module=MODULE_FACES)
    active = bool(latest and latest.status == "running" and latest.phase == "clustering")
    last_error = latest.last_error if latest and latest.status == "failed" else None
    return {
        "active": active,
        "last_error": last_error,
        "last_run_at": latest.finished_at if latest else None,
    }


@router.post("/cluster", status_code=202)
async def cluster_people(request: Request, *, regroup: bool = False) -> JSONResponse:
    """Start a clustering-only run tracked in index_runs (phase=clustering)."""
    executor = getattr(request.app.state, "index_executor", None)
    if executor is None:
        raise HTTPException(status_code=503, detail="index executor not ready")
    try:
        run_id = await executor.start_cluster_run(regroup=regroup)
    except IndexRunConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        status = 503 if "Qdrant" in str(exc) or "qdrant" in str(exc) else 400
        raise HTTPException(status_code=status, detail=str(exc)) from exc

    return JSONResponse(
        status_code=202,
        content={"run_id": run_id, "regroup": regroup, "module": MODULE_FACES},
    )
