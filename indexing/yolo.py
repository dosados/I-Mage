import logging
from pathlib import Path

from db.database import Database
from ml.objects.base import ObjectsRetriever

logger = logging.getLogger(__name__)


def index_yolo_image(
    db: Database,
    path: Path,
    retriever: ObjectsRetriever,
    *,
    model_version: str,
) -> None:
    record = db.images.get_by_path(path)
    if record is None:
        raise ValueError(f"image not registered: {path}")

    db.image_yolo.mark_running(record.id, model_version=model_version)
    try:
        detections = retriever.detect(path)
        db.detections.replace_for_image(record.id, detections)
        db.image_yolo.mark_done(record.id, model_version=model_version)
    except Exception as exc:
        db.image_yolo.mark_failed(record.id, str(exc), model_version=model_version)
        raise


def index_yolo_gap(
    db: Database,
    paths: list[Path],
    retriever: ObjectsRetriever,
    *,
    model_version: str,
) -> None:
    for path in paths:
        try:
            index_yolo_image(db, path, retriever, model_version=model_version)
        except Exception:
            logger.exception("failed to index yolo for %s", path)
