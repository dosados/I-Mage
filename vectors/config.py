import os

CLIP_VECTOR_DIM = 512
FACE_VECTOR_DIM = 512

CONTEXT_COLLECTION = "context"
FACES_COLLECTION = "faces"

DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"


def resolve_qdrant_url(url: str | None = None) -> str:
    """Resolve the Qdrant server URL.

    Qdrant runs as a standalone server only (see docker-compose.yml). Embedded
    file mode was removed because its single-writer lock made multi-process /
    --reload setups fail with "already accessed" and desynced the index.
    """
    if url is not None:
        return url
    return os.environ.get("QDRANT_URL", DEFAULT_QDRANT_URL)
