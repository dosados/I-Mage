"""COCO object keywords for unified search (YOLO class labels)."""

from __future__ import annotations

import re

COCO_CLASSES: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

COCO_CLASS_SET = frozenset(COCO_CLASSES)

# Longest labels first so "traffic light" wins over "light".
_COCO_BY_LENGTH = sorted(COCO_CLASSES, key=len, reverse=True)

_KEYWORD_GROUPS: dict[str, tuple[str, ...]] = {
    "Люди": ("person",),
    "Транспорт": (
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
    ),
    "Животные": (
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    ),
    "Еда": (
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
    ),
    "Мебель и интерьер": (
        "chair",
        "couch",
        "bed",
        "dining table",
        "potted plant",
        "toilet",
        "tv",
        "clock",
        "vase",
    ),
    "Электроника": (
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "refrigerator",
    ),
    "Улица": (
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
    ),
    "Спорт и отдых": (
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
    ),
    "Сумки и аксессуары": (
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
    ),
    "Посуда": (
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "sink",
    ),
    "Прочее": (
        "book",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ),
}


def normalize_label(label: str) -> str:
    return label.strip().lower()


def is_valid_label(label: str) -> bool:
    return normalize_label(label) in COCO_CLASS_SET


def extract_labels_from_query(query: str) -> list[str]:
    """Find COCO class names mentioned in *query* (word-boundary aware)."""
    lowered = query.lower()
    found: list[str] = []
    consumed: list[tuple[int, int]] = []

    for label in _COCO_BY_LENGTH:
        pattern = r"(?<!\w)" + re.escape(label) + r"(?!\w)"
        for match in re.finditer(pattern, lowered):
            start, end = match.span()
            if any(not (end <= s or start >= e) for s, e in consumed):
                continue
            consumed.append((start, end))
            found.append(label)

    # Preserve order of appearance in the query.
    found.sort(key=lambda lbl: lowered.find(lbl))
    return found


def merge_labels(*sources: list[str] | None) -> list[str]:
    """Deduplicate labels while preserving first-seen order."""
    seen: set[str] = set()
    merged: list[str] = []
    for source in sources:
        if not source:
            continue
        for raw in source:
            label = normalize_label(raw)
            if not label or label in seen:
                continue
            if label not in COCO_CLASS_SET:
                continue
            seen.add(label)
            merged.append(label)
    return merged


def keywords_payload() -> dict:
    return {
        "classes": list(COCO_CLASSES),
        "groups": {name: list(labels) for name, labels in _KEYWORD_GROUPS.items()},
    }
