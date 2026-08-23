import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_METADATA = ROOT / "data" / "celeba" / "metadata.json"
DEFAULT_SOURCE_IMAGES = ROOT / "data" / "celeba" / "img_align_celeba"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "small_celeba"
DEFAULT_TARGET_COUNT = 5_000


def make_record(filename: str, identity_id: int, image_path: Path) -> dict:
    return {
        "filename": filename,
        "identity_id": identity_id,
        "path": str(image_path.relative_to(ROOT)),
    }


def load_source_records(metadata_path: Path) -> list[dict]:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return payload["images"]


def group_by_identity(records: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        grouped[int(record["identity_id"])].append(record)
    return grouped


def select_identities(
    grouped: dict[int, list[dict]],
    *,
    target_count: int,
) -> list[int]:
    ordered = sorted(
        grouped,
        key=lambda identity_id: len(grouped[identity_id]),
        reverse=True,
    )

    selected: list[int] = []
    total = 0
    for identity_id in ordered:
        selected.append(identity_id)
        total += len(grouped[identity_id])
        if total >= target_count:
            break

    if not selected:
        raise RuntimeError("no identities selected")

    return selected


def build_small_celeba(
    *,
    source_metadata: Path,
    source_images: Path,
    output_dir: Path,
    target_count: int,
) -> None:
    if not source_metadata.is_file():
        raise FileNotFoundError(f"source metadata not found: {source_metadata}")
    if not source_images.is_dir():
        raise NotADirectoryError(f"source images directory not found: {source_images}")

    records = load_source_records(source_metadata)
    grouped = group_by_identity(records)
    selected_identities = select_identities(grouped, target_count=target_count)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    copied_records: list[dict] = []
    for identity_id in selected_identities:
        for record in sorted(grouped[identity_id], key=lambda item: item["filename"]):
            source_path = source_images / record["filename"]
            if not source_path.is_file():
                raise FileNotFoundError(f"source image not found: {source_path}")

            destination = output_dir / record["filename"]
            shutil.copy2(source_path, destination)
            copied_records.append(
                make_record(record["filename"], identity_id, destination),
            )

    identity_ids = {record["identity_id"] for record in copied_records}
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "dataset": "small_celeba",
                "source": "CelebA",
                "count": len(copied_records),
                "identities": len(identity_ids),
                "images": copied_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"output: {output_dir}")
    print(f"images: {len(copied_records)}")
    print(f"identities: {len(identity_ids)}")
    print(f"metadata: {metadata_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a smaller CelebA subset for fast face-search testing",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=DEFAULT_TARGET_COUNT,
        help="Approximate number of images to include (full identities only)",
    )
    parser.add_argument("--source-metadata", type=Path, default=DEFAULT_SOURCE_METADATA)
    parser.add_argument("--source-images", type=Path, default=DEFAULT_SOURCE_IMAGES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.target_count < 1:
        print("target-count must be >= 1", file=sys.stderr)
        return 1

    try:
        build_small_celeba(
            source_metadata=args.source_metadata,
            source_images=args.source_images,
            output_dir=args.output_dir,
            target_count=args.target_count,
        )
    except Exception as exc:
        print(f"failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
