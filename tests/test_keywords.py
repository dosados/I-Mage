from __future__ import annotations

import pytest

from api.keywords import (
    COCO_CLASSES,
    extract_labels_from_query,
    is_valid_label,
    merge_labels,
    normalize_label,
)


class TestKeywordExtraction:
    def test_single_word(self) -> None:
        assert extract_labels_from_query("sunset airplane") == ["airplane"]

    def test_multi_word_class(self) -> None:
        assert extract_labels_from_query("near traffic light") == ["traffic light"]

    def test_multiple_labels_preserve_order(self) -> None:
        assert extract_labels_from_query("dog and cat") == ["dog", "cat"]

    def test_no_partial_substring_match(self) -> None:
        assert extract_labels_from_query("airplanes") == []

    def test_case_insensitive(self) -> None:
        assert extract_labels_from_query("AirPlane") == ["airplane"]

    def test_no_keywords(self) -> None:
        assert extract_labels_from_query("закат на море") == []


class TestMergeLabels:
    def test_dedup_preserves_order(self) -> None:
        assert merge_labels(["dog", "cat"], ["cat", "airplane"]) == [
            "dog",
            "cat",
            "airplane",
        ]

    def test_invalid_filtered(self) -> None:
        assert merge_labels(["unicorn"], ["dog"]) == ["dog"]

    def test_normalize(self) -> None:
        assert merge_labels([" Dog "]) == ["dog"]


class TestKeywordCatalog:
    def test_all_classes_valid(self) -> None:
        for label in COCO_CLASSES:
            assert is_valid_label(label)
            assert normalize_label(label) == label
