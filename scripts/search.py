import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.embeddings import CLIPEmbeddingModel

DEFAULT_DATA_DIR = ROOT / "data" / "flickr30k"
METADATA_FILE = "metadata.json"


def load_metadata(data_dir: Path) -> list[dict]:
    metadata_path = data_dir / METADATA_FILE
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata not found: {metadata_path}. Run scripts/download_data.py first."
        )

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return payload["images"]


def build_embeddings(
    model: CLIPEmbeddingModel,
    images: list[dict],
) -> tuple[list[str], np.ndarray]:
    paths: list[str] = []
    vectors: list[np.ndarray] = []

    for index, item in enumerate(images):
        image_path = ROOT / item["path"]
        embedding = model.encode_image(image_path)
        paths.append(item["path"])
        vectors.append(embedding)
        print(f"[{index + 1}/{len(images)}] embedded {item['filename']}")

    return paths, np.stack(vectors, axis=0)


def find_best_match(
    query: str,
    model: CLIPEmbeddingModel,
    paths: list[str],
    embeddings: np.ndarray,
    images: list[dict],
) -> dict:
    query_embedding = model.encode_text(query)
    scores = embeddings @ query_embedding
    best_index = int(np.argmax(scores))

    path_by_relpath = {item["path"]: item for item in images}
    best_path = paths[best_index]
    item = path_by_relpath[best_path]

    return {
        "query": query,
        "path": best_path,
        "absolute_path": str(ROOT / best_path),
        "score": float(scores[best_index]),
        "image_id": item["id"],
        "captions_en": item["captions_en"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Find best image match for a text query")
    parser.add_argument("query", help="Text search query")
    parser.add_argument("--limit", type=int, default=None, help="Search only among first N images")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    images = load_metadata(args.data_dir)
    if args.limit is not None:
        if args.limit < 1:
            print("limit must be >= 1", file=sys.stderr)
            return 1
        images = images[: args.limit]

    if not images:
        print("no images found in metadata", file=sys.stderr)
        return 1

    model = CLIPEmbeddingModel()
    paths, embeddings = build_embeddings(model, images)

    result = find_best_match(args.query, model, paths, embeddings, images)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
