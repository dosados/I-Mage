from __future__ import annotations

from pathlib import Path

import pytest

from api.paths import resolve_image_paths, resolve_scoped_image_paths
from helpers import make_image_file
from io_utils.fs import IMAGE_SUFFIXES, collect_files, filter_image_paths, reveal_in_file_manager
from io_utils.scan import collect_scoped_files


class TestFsUtils:
    def test_collect_files_recursive(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        make_image_file(tmp_path, "root.jpg")
        make_image_file(nested, "deep.png")
        (tmp_path / "notes.txt").write_text("nope")
        paths = collect_files(tmp_path, IMAGE_SUFFIXES, recursive=True)
        assert {p.name for p in paths} == {"root.jpg", "deep.png"}

    def test_collect_files_non_recursive(self, tmp_path: Path) -> None:
        nested = tmp_path / "sub"
        nested.mkdir()
        make_image_file(tmp_path, "root.jpg")
        make_image_file(nested, "deep.jpg")
        paths = collect_files(tmp_path, IMAGE_SUFFIXES, recursive=False)
        assert [p.name for p in paths] == ["root.jpg"]

    def test_collect_files_missing_dir(self, tmp_path: Path) -> None:
        with pytest.raises(NotADirectoryError):
            collect_files(tmp_path / "missing", IMAGE_SUFFIXES)

    def test_filter_image_paths(self, tmp_path: Path, caplog) -> None:
        good = make_image_file(tmp_path, "ok.webp")
        bad = tmp_path / "x.txt"
        bad.write_text("x")
        missing = tmp_path / "gone.jpg"
        result = filter_image_paths([good, bad, missing])
        assert result == [good]

    def test_reveal_opens_folder(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        photo = make_image_file(tmp_path, "shot.jpg")
        calls: list[list[str]] = []

        def fake_which(name: str) -> str | None:
            return "/usr/bin/xdg-open" if name == "xdg-open" else None

        def fake_popen(command, **_kwargs):
            calls.append(list(command))
            return None

        monkeypatch.setattr("io_utils.fs.shutil.which", fake_which)
        monkeypatch.setattr("io_utils.fs.sys.platform", "linux")
        monkeypatch.setattr("io_utils.fs.subprocess.Popen", fake_popen)
        assert reveal_in_file_manager(photo) is True
        assert calls == [["xdg-open", str(photo.parent)]]
        assert reveal_in_file_manager(tmp_path / "missing.jpg") is False


class TestScopedScan:
    def test_collect_scoped_dedupes_and_ignores(self, tmp_path: Path) -> None:
        left = tmp_path / "left"
        right = tmp_path / "right"
        left.mkdir()
        right.mkdir()
        shared_name = "same.jpg"
        make_image_file(left, shared_name, b"L")
        make_image_file(right, shared_name, b"R")
        make_image_file(left, "keep.jpg")
        make_image_file(left, "skip.tmp.jpg")  # still .jpg suffix — ignore by name pattern

        # Create a file that matches ignore glob by path segment.
        cache = left / "cache"
        cache.mkdir()
        make_image_file(cache, "thumb.jpg")

        paths = collect_scoped_files(
            [left, right, left],  # duplicate include
            ignore_globs=["**/cache/**"],
        )
        names = {p.name for p in paths}
        assert "keep.jpg" in names
        assert "thumb.jpg" not in names
        # Two different absolute paths with same basename both kept.
        assert sum(1 for p in paths if p.name == shared_name) == 2

    def test_resolve_scoped_empty_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="no images found"):
            resolve_scoped_image_paths([str(empty)])

    def test_resolve_image_paths_limit(self, tmp_path: Path) -> None:
        for i in range(5):
            make_image_file(tmp_path, f"{i}.jpg")
        paths = resolve_image_paths(tmp_path, limit=2)
        assert len(paths) == 2

    def test_ignore_by_basename_glob(self, tmp_path: Path) -> None:
        make_image_file(tmp_path, "photo.jpg")
        make_image_file(tmp_path, "secret.jpg")
        paths = collect_scoped_files([tmp_path], ignore_globs=["secret.jpg"])
        assert [p.name for p in paths] == ["photo.jpg"]
