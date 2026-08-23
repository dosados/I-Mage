import bisect
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ml.embeddings.base import EmbeddingModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageMatch:
    path: Path
    score: float


@dataclass(frozen=True)
class SearchResult:
    query: str
    matches: list[ImageMatch]


def search_by_description(
    query: str,
    paths: Iterable[str | Path],
    model: EmbeddingModel,
    *,
    k: int = 1,
) -> SearchResult:
    image_paths = [Path(path) for path in paths]
    if not image_paths:
        raise ValueError("no image paths provided")

    responses_list = []
    query_embedding = model.encode_text(query)

    for image_path in image_paths:
        try:
            embedding = model.encode_image(image_path)
        except Exception:
            logger.exception("failed to embed image, skipping: %s", image_path)
            continue

        score = float(embedding @ query_embedding)

        item = ImageMatch(path=image_path, score=score)
        bisect.insort(responses_list, item, key=lambda x: -x.score)
        if len(responses_list) > k:
            responses_list.pop()

    if len(responses_list) == 0:
        raise ValueError("no valid pictures in provided paths")

    return SearchResult(query=query, matches=responses_list)
