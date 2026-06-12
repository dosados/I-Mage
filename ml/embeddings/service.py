import bisect
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from io_utils.fs import IMAGE_SUFFIXES, collect_files

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
    directory: Path,
    model: EmbeddingModel,
    *,
    suffixes: Iterable[str] | None = None,
    recursive: bool = True,
    limit: int | None = None,
    k: int = 1,
) -> SearchResult:
    image_paths = collect_files(
        directory,
        suffixes or IMAGE_SUFFIXES,
        recursive=recursive,
    )
    if limit is not None:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise ValueError(f"no images found in {directory}")

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
        raise ValueError("no valid pictures in directory")

    return SearchResult(query=query, matches=responses_list)
