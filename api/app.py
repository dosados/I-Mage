import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from api.schemas import ImageMatch, SearchRequest, SearchResponse
from ml.embeddings import CLIPEmbeddingModel
from ml.embeddings.service import search_by_description

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEARCH_DIR = ROOT / "data" / "flickr30k" / "images"
WEB_DIR = ROOT / "clients" / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = CLIPEmbeddingModel()
    app.state.search_dir = Path(os.environ.get("IMAGE_SEARCH_DIR", DEFAULT_SEARCH_DIR))
    yield


app = FastAPI(title="I-Mage", version="0.1.0", lifespan=lifespan)


@app.get("/")
def web_index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/images/{filename}")
def get_image(filename: str, request: Request) -> FileResponse:
    safe_name = Path(filename).name
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="invalid filename")

    image_path = request.app.state.search_dir / safe_name
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="image not found")

    return FileResponse(image_path)


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    if not hasattr(request.app.state, "model"):
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest, http_request: Request) -> SearchResponse:
    try:
        result = search_by_description(
            request.query,
            http_request.app.state.search_dir,
            http_request.app.state.model,
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
