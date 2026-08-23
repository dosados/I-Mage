import logging
from pathlib import Path

from db.database import Database
from ml.embeddings.base import EmbeddingModel
from vectors.store import VectorStore

logger = logging.getLogger(__name__)


def index_clip_image(
    db: Database,
    vector_store: VectorStore,
    path: Path,
    model: EmbeddingModel,
    *,
    model_version: str,
) -> None:
    record = db.images.get_by_path(path)
    if record is None:
        raise ValueError(f"image not registered: {path}")

    db.image_clip.mark_running(record.id, model_version=model_version)
    try:
        embedding = model.encode_image(path)
        vector_store.upsert_context(record.id, embedding, model_version=model_version)
        db.image_clip.mark_done(record.id, model_version=model_version)
    except Exception as exc:
        db.image_clip.mark_failed(record.id, str(exc), model_version=model_version)
        raise


def index_clip_gap(
    db: Database,
    vector_store: VectorStore,
    paths: list[Path],
    model: EmbeddingModel,
    *,
    model_version: str,
) -> None:
    for path in paths:
        try:
            index_clip_image(
                db,
                vector_store,
                path,
                model,
                model_version=model_version,
            )
        except Exception:
            logger.exception("failed to index clip for %s", path)
