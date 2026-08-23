import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"
CELEBA_DIR_NAME = "celeba"
IMAGES_DIR_NAME = "img_align_celeba"
METADATA_NAME = "metadata.json"
EXPECTED_IMAGE_COUNT = 202_599
EXPECTED_IDENTITY_COUNT = 10_177


def require_gdown() -> None:
    try:
        import gdown  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "gdown is required for CelebA download. Install it with: pip install gdown"
        ) from exc


def celeba_images_dir(data_dir: Path) -> Path:
    return data_dir / CELEBA_DIR_NAME / IMAGES_DIR_NAME


def make_record(filename: str, identity_id: int, image_path: Path) -> dict:
    return {
        "filename": filename,
        "identity_id": identity_id,
        "path": str(image_path.relative_to(ROOT)),
    }


def download_celeba(data_dir: Path) -> None:
    from torchvision.datasets import CelebA

    require_gdown()
    data_dir.mkdir(parents=True, exist_ok=True)

    print("downloading CelebA (torchvision + identity labels)...")
    dataset = CelebA(
        root=str(data_dir),
        split="all",
        target_type="identity",
        download=True,
    )

    images_dir = celeba_images_dir(data_dir)
    if not images_dir.is_dir():
        raise RuntimeError(f"downloaded CelebA images not found at {images_dir}")

    records = [
        make_record(
            dataset.filename[index],
            int(dataset.identity[index]),
            images_dir / dataset.filename[index],
        )
        for index in range(len(dataset))
    ]

    metadata_path = data_dir / CELEBA_DIR_NAME / METADATA_NAME
    identity_ids = {record["identity_id"] for record in records}
    metadata_path.write_text(
        json.dumps(
            {
                "dataset": "CelebA",
                "count": len(records),
                "identities": len(identity_ids),
                "images": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"images: {images_dir}")
    print(f"metadata: {metadata_path}")
    print(f"done: {len(records)} images, {len(identity_ids)} identities")

    errors: list[str] = []
    image_files = sorted(images_dir.glob("*.jpg"))
    if len(image_files) != len(records):
        errors.append(
            f"image files ({len(image_files)}) != metadata records ({len(records)})"
        )
    if len(records) != EXPECTED_IMAGE_COUNT:
        errors.append(f"expected {EXPECTED_IMAGE_COUNT} records, got {len(records)}")
    if len(identity_ids) != EXPECTED_IDENTITY_COUNT:
        errors.append(
            f"expected {EXPECTED_IDENTITY_COUNT} identities, got {len(identity_ids)}"
        )

    if errors:
        print("validation issues:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        raise RuntimeError("CelebA validation failed")

    print("validation: ok")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download CelebA with identity labels into data/celeba/",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root data directory (images go to <data-dir>/celeba/img_align_celeba/)",
    )
    args = parser.parse_args()

    try:
        download_celeba(args.data_dir)
    except Exception as exc:
        print(f"failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
