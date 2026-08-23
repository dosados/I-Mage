import logging
from pathlib import Path

from db.database import Database
from ml.faces.base import FaceRecognizer
from vectors.store import VectorStore

logger = logging.getLogger(__name__)


def store_faces(
    db: Database,
    vector_store: VectorStore,
    path: Path,
    faces: list,
    *,
    model_version: str,
) -> None:
    """Persist already-analyzed faces (DB rows + Qdrant vectors).

    Split out from analysis so the CPU/GPU-heavy ``recognizer.analyze`` can run
    in a worker-thread pool while the DB/Qdrant writes stay on one thread.
    """
    record = db.images.get_by_path(path)
    if record is None:
        raise ValueError(f"image not registered: {path}")

    # Face ids are deterministic (uuid5 of image+bbox+model), so re-detecting the
    # same image just re-upserts the same points. Only faces that DISAPPEARED need
    # an explicit delete — avoiding a per-image filtered delete keeps re-index fast.
    prior_face_ids = {face.id for face in db.faces.list_for_image(record.id)}

    db.image_faces.mark_running(record.id, model_version=model_version)
    try:
        face_records = db.faces.replace_for_image(
            record.id,
            faces,
            model_version=model_version,
        )
        face_vectors = [
            (face_record.id, face_record.image_id, ml_face.embedding)
            for face_record, ml_face in zip(face_records, faces, strict=True)
        ]
        # Vectors go to Qdrant BEFORE mark_done so a crash can never leave the DB
        # claiming DONE without the vectors existing (gap re-index self-heals).
        vector_store.upsert_faces(face_vectors, model_version=model_version)
        removed = prior_face_ids - {face_record.id for face_record in face_records}
        if removed:
            vector_store.delete_face_points(removed)
        db.image_faces.mark_done(record.id, model_version=model_version)
    except Exception as exc:
        db.image_faces.mark_failed(record.id, str(exc), model_version=model_version)
        raise


def index_faces_image(
    db: Database,
    vector_store: VectorStore,
    path: Path,
    recognizer: FaceRecognizer,
    *,
    model_version: str,
) -> None:
    faces = recognizer.analyze(path)
    store_faces(db, vector_store, path, faces, model_version=model_version)


def index_faces_gap(
    db: Database,
    vector_store: VectorStore,
    paths: list[Path],
    recognizer: FaceRecognizer,
    *,
    model_version: str,
) -> None:
    for path in paths:
        try:
            index_faces_image(
                db,
                vector_store,
                path,
                recognizer,
                model_version=model_version,
            )
        except Exception:
            logger.exception("failed to index faces for %s", path)
