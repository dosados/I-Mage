import argparse
import gc
import json
import sys
import time
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "flickr30k"
DEFAULT_DATASET = "ai-enthusiasm-community/Flickr30k"
EXPECTED_COUNT = 31_783
METADATA_SAVE_EVERY = 500
MAX_RETRIES = 20
RETRY_DELAY_SEC = 10


def save_metadata(metadata_path: Path, dataset_name: str, records_by_id: dict[str, dict]) -> None:
    records = [records_by_id[image_id] for image_id in sorted(records_by_id)]
    metadata_path.write_text(
        json.dumps(
            {"dataset": dataset_name, "count": len(records), "images": records},
            indent=2,
        ),
        encoding="utf-8",
    )


def load_existing_records(metadata_path: Path) -> dict[str, dict]:
    if not metadata_path.exists():
        return {}

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {item["id"]: item for item in payload["images"]}


def make_record(image_id: str, image_path: Path, captions_en: list[str]) -> dict:
    return {
        "id": image_id,
        "filename": f"{image_id}.jpg",
        "path": str(image_path.relative_to(ROOT)),
        "captions_en": captions_en,
    }


def validate_dataset(data_dir: Path, records_by_id: dict[str, dict]) -> list[str]:
    images_dir = data_dir / "images"
    errors: list[str] = []

    image_files = sorted(images_dir.glob("*.jpg"))
    if len(image_files) != len(records_by_id):
        errors.append(
            f"image files ({len(image_files)}) != metadata records ({len(records_by_id)})"
        )

    for image_path in image_files:
        image_id = image_path.stem
        if image_id not in records_by_id:
            errors.append(f"missing metadata for {image_path.name}")
            continue
        if records_by_id[image_id]["path"] != str(image_path.relative_to(ROOT)):
            errors.append(f"path mismatch for {image_id}")

    if len(records_by_id) != EXPECTED_COUNT:
        errors.append(f"expected {EXPECTED_COUNT} records, got {len(records_by_id)}")

    return errors


def download_flickr30k(
    data_dir: Path,
    dataset_name: str,
    limit: int | None,
    resume: bool,
) -> None:
    images_dir = data_dir / "images"
    metadata_path = data_dir / "metadata.json"
    images_dir.mkdir(parents=True, exist_ok=True)

    records_by_id = load_existing_records(metadata_path) if resume else {}
    existing_files = {path.stem for path in images_dir.glob("*.jpg")}

    attempt = 0
    while attempt < MAX_RETRIES:
        attempt += 1
        try:
            dataset = load_dataset(dataset_name, split="train", streaming=True)
            downloaded = 0
            repaired = 0
            skipped = 0

            for index, row in enumerate(dataset):
                if limit is not None and index >= limit:
                    break

                image_id = row["image_uid"]
                image_path = images_dir / f"{image_id}.jpg"
                file_exists = image_id in existing_files

                if file_exists and image_id in records_by_id:
                    skipped += 1
                    continue

                if not file_exists:
                    row["image"].convert("RGB").save(image_path, format="JPEG")
                    existing_files.add(image_id)
                    downloaded += 1
                    print(f"[{index + 1}] downloaded {image_path.name}")
                else:
                    repaired += 1
                    print(f"[{index + 1}] repaired metadata for {image_path.name}")

                records_by_id[image_id] = make_record(
                    image_id,
                    image_path,
                    row["caption_en"],
                )

                if len(records_by_id) % METADATA_SAVE_EVERY == 0:
                    save_metadata(metadata_path, dataset_name, records_by_id)
                    print(f"checkpoint: {len(records_by_id)} records in metadata")

            del dataset
            gc.collect()

            save_metadata(metadata_path, dataset_name, records_by_id)
            print(
                f"done: {len(records_by_id)} records "
                f"(downloaded={downloaded}, repaired={repaired}, skipped={skipped})"
            )

            errors = validate_dataset(data_dir, records_by_id)
            if errors:
                print("validation issues:", file=sys.stderr)
                for error in errors:
                    print(f"  - {error}", file=sys.stderr)
                if len(records_by_id) < (limit or EXPECTED_COUNT):
                    raise RuntimeError("dataset incomplete after pass")
                return

            print("validation: ok")
            return

        except Exception as exc:
            save_metadata(metadata_path, dataset_name, records_by_id)
            print(f"attempt {attempt}/{MAX_RETRIES} failed: {exc}", file=sys.stderr)
            if attempt >= MAX_RETRIES:
                raise
            time.sleep(RETRY_DELAY_SEC)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Flickr30k into data/")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap for debugging (default: full dataset)",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start from scratch (ignores existing files/metadata)",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        print("limit must be >= 1", file=sys.stderr)
        return 1

    download_flickr30k(args.data_dir, args.dataset, args.limit, resume=not args.no_resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
