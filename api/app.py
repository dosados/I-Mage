import json
import logging
import os
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from api.index import router as index_router
from api.people import router as people_router
from api.schemas import (
    ClassImageMatch,
    ClassSearchRequest,
    ClassSearchResponse,
    FaceEmbedResponse,
    FaceImageMatch,
    FaceSearchRequest,
    FaceSearchResponse,
    ImageMatch,
    SearchRequest,
    SearchResponse,
    RevealFileRequest,
    UnifiedMatchResponse,
    UnifiedSearchRequest,
    UnifiedSearchResponse,
)
from api.search import (
    iter_class_search_events,
    iter_description_search_events,
    iter_face_search_events,
    iter_unified_search_events,
    run_embed_query_face,
    run_search_by_class,
    run_search_by_description,
    run_search_by_face,
    run_unified_search,
)
from api.settings import router as settings_router
from api.uploads import read_upload_image
from api.keywords import keywords_payload
from io_utils.fs import reveal_in_file_manager
from db.database import Database
from db.scan_config import default_scan_config
from indexing.background_indexer import background_indexer_loop
from indexing.executor import IndexExecutor
from indexing.runner import IndexModels
from ml.embeddings import CLIPEmbeddingModel
from ml.faces import ArcFaceRecognizer
from ml.objects import YoloObjectsRetriever
from vectors import create_vector_store

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEARCH_DIR = ROOT / "data" / "flickr30k" / "images"
DEFAULT_FACE_SEARCH_DIR = ROOT / "data" / "small_celeba"
WEB_DIR = ROOT / "clients" / "web"
logger = logging.getLogger(__name__)


def resolve_image_path(request: Request, filename: str) -> Path:
    safe_name = Path(filename).name
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="invalid filename")

    with request.app.state.db:
        config = request.app.state.db.get_scan_config()
    for directory in config.include_directories:
        image_path = Path(directory) / safe_name
        if image_path.is_file():
            return image_path

    for directory in (
        request.app.state.search_dir,
        request.app.state.face_search_dir,
    ):
        image_path = directory / safe_name
        if image_path.is_file():
            return image_path

    raise HTTPException(status_code=404, detail="image not found")


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    logger.info("loading CLIP…")
    app.state.model = CLIPEmbeddingModel()
    logger.info("loading YOLO…")
    app.state.objects_model = YoloObjectsRetriever()
    logger.info("loading ArcFace…")
    app.state.face_model = ArcFaceRecognizer()
    logger.info("connecting Qdrant + opening DB…")
    app.state.vector_store = create_vector_store()
    app.state.db = Database(vector_store=app.state.vector_store)
    app.state.search_dir = Path(os.environ.get("IMAGE_SEARCH_DIR", DEFAULT_SEARCH_DIR))
    app.state.face_search_dir = Path(
        os.environ.get("FACE_SEARCH_DIR", DEFAULT_FACE_SEARCH_DIR)
    )
    with app.state.db:
        app.state.db.set_default_scan_config(
            default_scan_config(
                search_dir=app.state.search_dir,
                face_search_dir=app.state.face_search_dir,
            )
        )
        stale = app.state.db.index_runs.fail_stale_runs()
        if stale:
            logger.info("marked %d stale index runs as failed", stale)

    indexer_models = IndexModels(
        clip=app.state.model,
        yolo=app.state.objects_model,
        faces=app.state.face_model,
    )
    app.state.index_executor = IndexExecutor(
        db=app.state.db,
        vector_store=app.state.vector_store,
        models=indexer_models,
    )

    stop_event = asyncio.Event()
    indexer_task = asyncio.create_task(
        background_indexer_loop(
            db=app.state.db,
            vector_store=app.state.vector_store,
            models=indexer_models,
            stop_event=stop_event,
            executor=app.state.index_executor,
        )
    )
    app.state.background_indexer_stop = stop_event
    app.state.background_indexer_task = indexer_task
    logger.info("startup complete — listening")

    yield

    stop_event.set()
    await indexer_task
    await app.state.index_executor.shutdown()
    app.state.db.close()
    app.state.vector_store.close()


app = FastAPI(title="I-Mage", version="0.1.0", lifespan=lifespan)
app.include_router(settings_router)
app.include_router(index_router)
app.include_router(people_router)


@app.get("/")
def web_index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/images/by-id/{image_id}")
def get_image_by_id(image_id: str, request: Request) -> FileResponse:
    """Serve an image by its catalog id.

    Person thumbnails use this instead of a bare filename so images with the
    same basename in different directories don't collide onto the wrong file.
    """
    with request.app.state.db:
        record = request.app.state.db.images.get_by_id(image_id)
    if record is None:
        raise HTTPException(status_code=404, detail="image not found")
    path = Path(record.path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="image file missing")
    return FileResponse(path)


@app.get("/images/{filename}")
def get_image(filename: str, request: Request) -> FileResponse:
    return FileResponse(resolve_image_path(request, filename))


@app.post("/files/reveal")
def reveal_file(payload: RevealFileRequest, request: Request) -> dict[str, object]:
    """Open the catalog image in the desktop file manager, or at least return its path."""
    with request.app.state.db:
        record = request.app.state.db.images.get_by_id(payload.image_id)
    if record is None:
        raise HTTPException(status_code=404, detail="image not found")
    path = Path(record.path)
    opened = reveal_in_file_manager(path) if path.exists() else False
    return {"path": str(path), "opened": opened, "exists": path.exists()}


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    if not hasattr(request.app.state, "model"):
        raise HTTPException(status_code=503, detail="model not loaded")
    if not hasattr(request.app.state, "objects_model"):
        raise HTTPException(status_code=503, detail="objects model not loaded")
    if not hasattr(request.app.state, "face_model"):
        raise HTTPException(status_code=503, detail="face model not loaded")
    store = request.app.state.vector_store
    qdrant_status = "ok" if store.available else "unavailable"
    result = {"status": "ok", "qdrant": qdrant_status}
    if not store.available and store.last_error:
        result["qdrant_error"] = store.last_error
    return result


@app.get("/keywords")
def list_keywords() -> dict:
    return keywords_payload()


@app.post("/search/unified", response_model=UnifiedSearchResponse)
def search_unified(
    request: UnifiedSearchRequest,
    http_request: Request,
) -> UnifiedSearchResponse:
    try:
        result = run_unified_search(
            request.query,
            http_request.app.state.model,
            http_request.app.state.objects_model,
            http_request.app.state.db,
            http_request.app.state.vector_store,
            labels=request.labels,
            limit=request.limit,
            k=request.k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return UnifiedSearchResponse(
        query=result.query,
        labels=result.labels,
        matches=[
            UnifiedMatchResponse(
                path=str(match.path),
                clip_score=match.clip_score,
                yolo=match.yolo,
                sources=match.sources,
                rank_score=match.rank_score,
            )
            for match in result.matches
        ],
    )


@app.post("/search/unified/stream")
def search_unified_stream(
    request: UnifiedSearchRequest,
    http_request: Request,
) -> StreamingResponse:
    def event_stream() -> Iterator[bytes]:
        for event in iter_unified_search_events(
            request.query,
            http_request.app.state.model,
            http_request.app.state.objects_model,
            http_request.app.state.db,
            http_request.app.state.vector_store,
            labels=request.labels,
            limit=request.limit,
            k=request.k,
        ):
            yield (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest, http_request: Request) -> SearchResponse:
    try:
        result = run_search_by_description(
            request.query,
            http_request.app.state.model,
            http_request.app.state.db,
            http_request.app.state.vector_store,
            limit=request.limit,
            k=request.k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return SearchResponse(
        query=result.query,
        matches=[
            ImageMatch(path=str(match.path), score=match.score)
            for match in result.matches
        ],
    )


@app.post("/search/stream")
def search_stream(request: SearchRequest, http_request: Request) -> StreamingResponse:
    def event_stream() -> Iterator[bytes]:
        for event in iter_description_search_events(
            request.query,
            http_request.app.state.model,
            http_request.app.state.db,
            http_request.app.state.vector_store,
            limit=request.limit,
            k=request.k,
        ):
            yield (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.post("/search/class", response_model=ClassSearchResponse)
def search_by_class_endpoint(
    request: ClassSearchRequest,
    http_request: Request,
) -> ClassSearchResponse:
    try:
        result = run_search_by_class(
            request.label,
            http_request.app.state.objects_model,
            http_request.app.state.db,
            limit=request.limit,
            k=request.k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ClassSearchResponse(
        label=result.label,
        matches=[
            ClassImageMatch(path=str(match.path), confidence=match.confidence)
            for match in result.matches
        ],
    )


@app.post("/search/class/stream")
def search_by_class_stream(
    request: ClassSearchRequest,
    http_request: Request,
) -> StreamingResponse:
    def event_stream() -> Iterator[bytes]:
        for event in iter_class_search_events(
            request.label,
            http_request.app.state.objects_model,
            http_request.app.state.db,
            limit=request.limit,
            k=request.k,
        ):
            yield (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.post("/search/face/embed", response_model=FaceEmbedResponse)
async def embed_face(
    http_request: Request,
    file: UploadFile = File(...),
) -> FaceEmbedResponse:
    image = await read_upload_image(file)

    try:
        result = run_embed_query_face(image, http_request.app.state.face_model)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return FaceEmbedResponse(
        embedding=result.embedding.tolist(),
        embedding_dim=http_request.app.state.face_model.embedding_dim,
        detection_score=result.detection_score,
    )


@app.post("/search/face", response_model=FaceSearchResponse)
def search_by_face_endpoint(
    request: FaceSearchRequest,
    http_request: Request,
) -> FaceSearchResponse:
    try:
        result = run_search_by_face(
            request.embedding,
            http_request.app.state.face_model,
            http_request.app.state.db,
            http_request.app.state.vector_store,
            limit=request.limit,
            k=request.k,
            threshold=request.threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return FaceSearchResponse(
        matches=[
            FaceImageMatch(path=str(match.path), score=match.score)
            for match in result.matches
        ],
    )


@app.post("/search/face/stream")
def search_by_face_stream(
    request: FaceSearchRequest,
    http_request: Request,
) -> StreamingResponse:
    """NDJSON progress stream: reconcile → index_gap → search → done."""

    def event_stream() -> Iterator[bytes]:
        for event in iter_face_search_events(
            request.embedding,
            http_request.app.state.face_model,
            http_request.app.state.db,
            http_request.app.state.vector_store,
            limit=request.limit,
            k=request.k,
            threshold=request.threshold,
        ):
            yield (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.post("/search/face/upload", response_model=FaceSearchResponse)
async def search_by_face_upload(
    http_request: Request,
    file: UploadFile = File(...),
    limit: int | None = Form(default=None),
    k: int = Form(default=10),
    threshold: float = Form(default=0.4),
) -> FaceSearchResponse:
    if limit is not None and limit < 1:
        raise HTTPException(status_code=400, detail="limit must be >= 1")
    if k < 1 or k > 50:
        raise HTTPException(status_code=400, detail="k must be between 1 and 50")
    if threshold < 0.0 or threshold > 1.0:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1")

    image = await read_upload_image(file)

    try:
        query = run_embed_query_face(image, http_request.app.state.face_model)
        result = run_search_by_face(
            query.embedding,
            http_request.app.state.face_model,
            http_request.app.state.db,
            http_request.app.state.vector_store,
            limit=limit,
            k=k,
            threshold=threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return FaceSearchResponse(
        matches=[
            FaceImageMatch(path=str(match.path), score=match.score)
            for match in result.matches
        ],
    )
