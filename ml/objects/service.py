import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ml.objects.base import Detection, ObjectsRetriever

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageDetections:
    path: Path
    detections: list[Detection]


def detect_objects(
    paths: Iterable[str | Path],
    retriever: ObjectsRetriever,
) -> list[ImageDetections]:
    image_paths = [Path(path) for path in paths]
    if not image_paths:
        raise ValueError("no image paths provided")

    results: list[ImageDetections] = []

    for image_path in image_paths:
        try:
            detections = retriever.detect(image_path)
        except Exception:
            logger.exception("failed to detect objects, skipping: %s", image_path)
            continue

        results.append(ImageDetections(path=image_path, detections=detections))

    if not results:
        raise ValueError("no valid pictures in provided paths")

    return results
