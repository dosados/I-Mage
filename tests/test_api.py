from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from db.database import Database
from fakes import FakeClip, FakeFaces, FakeVectorStore, FakeYolo, unit_vec
from helpers import register_file
from indexing.faces import index_faces_image
from ml.faces.base import Face
from ml.objects.base import Detection


@pytest.fixture
def api_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Boot FastAPI with fake ML + in-memory vectors; isolated SQLite."""
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    db_path = tmp_path / "api.db"
    store = FakeVectorStore()
    clip = FakeClip()
    yolo = FakeYolo()
    faces = FakeFaces()

    monkeypatch.setenv("IMAGE_DB_PATH", str(db_path))
    monkeypatch.setenv("IMAGE_SEARCH_DIR", str(photo_dir))
    monkeypatch.setenv("FACE_SEARCH_DIR", str(photo_dir))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    monkeypatch.setattr("api.app.CLIPEmbeddingModel", lambda: clip)
    monkeypatch.setattr("api.app.YoloObjectsRetriever", lambda: yolo)
    monkeypatch.setattr("api.app.ArcFaceRecognizer", lambda: faces)
    monkeypatch.setattr("api.app.create_vector_store", lambda: store)

    from api.app import app

    with TestClient(app) as client:
        yield {
            "client": client,
            "dir": photo_dir,
            "store": store,
            "clip": clip,
            "yolo": yolo,
            "faces": faces,
            "db_path": db_path,
        }


def _save_rgb(directory: Path, name: str) -> Path:
    path = directory / name
    Image.new("RGB", (48, 48), (40, 80, 120)).save(path)
    return path.resolve()


class TestHealthAndSettings:
    def test_health_ok(self, api_env) -> None:
        response = api_env["client"].get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["qdrant"] == "ok"

    def test_scan_settings_roundtrip(self, api_env) -> None:
        client = api_env["client"]
        directory = str(api_env["dir"])
        payload = {
            "include_directories": [directory],
            "ignore_globs": ["**/tmp/**"],
            "background_indexer_enabled": False,
            "schedule_interval_days": 3,
            "background_modules": ["clip"],
        }
        put = client.put("/settings/scan", json=payload)
        assert put.status_code == 200
        got = client.get("/settings/scan")
        assert got.status_code == 200
        data = got.json()
        assert data["include_directories"] == [directory]
        assert data["ignore_globs"] == ["**/tmp/**"]
        assert data["schedule_interval_days"] == 3


class TestIndexStatusApi:
    def test_status_includes_module_runs_without_fs_walk(self, api_env) -> None:
        client = api_env["client"]
        response = client.get("/index/status?stats=true")
        assert response.status_code == 200
        data = response.json()
        assert "module_runs" in data
        assert set(data["module_runs"]) >= {"yolo", "clip", "faces"}
        assert set(data["modules"]) >= {"yolo", "clip", "faces"}
        assert data["scope_total"] >= 0

    def test_faces_ready_from_counts(self, api_env) -> None:
        client = api_env["client"]
        response = client.get("/index/faces-ready")
        assert response.status_code == 200
        body = response.json()
        assert body["ready"] is False
        assert body["done"] == 0


class TestSearchApi:
    def test_face_embed_and_stream_search(self, api_env) -> None:
        client = api_env["client"]
        photo_dir: Path = api_env["dir"]
        store: FakeVectorStore = api_env["store"]
        faces: FakeFaces = api_env["faces"]

        # Configure scan + index one face into store via DB the app owns.
        client.put(
            "/settings/scan",
            json={
                "include_directories": [str(photo_dir)],
                "ignore_globs": [],
                "background_indexer_enabled": False,
                "schedule_interval_days": 7,
                "background_modules": ["faces"],
            },
        )
        path = _save_rgb(photo_dir, "person.jpg")
        query = unit_vec(7)
        faces.set_faces(
            "person.jpg",
            [Face(bbox=(0, 0, 10, 10), detection_score=0.99, embedding=query)],
        )
        faces.set_faces(
            "upload",
            [Face(bbox=(0, 0, 10, 10), detection_score=0.99, embedding=query)],
        )

        db = Database(path=api_env["db_path"], vector_store=store)
        with db:
            register_file(db, path)
            index_faces_image(db, store, path, faces, model_version=faces.model_name)
        db.close()

        upload = api_env["dir"].parent / "query.jpg"
        Image.new("RGB", (32, 32), (1, 2, 3)).save(upload)

        with upload.open("rb") as handle:
            embed = client.post(
                "/search/face/embed",
                files={"file": ("query.jpg", handle, "image/jpeg")},
            )
        assert embed.status_code == 200, embed.text
        embedding = embed.json()["embedding"]

        faces.analyze_calls.clear()
        stream = client.post(
            "/search/face/stream",
            json={"embedding": embedding, "k": 5, "threshold": 0.0},
        )
        assert stream.status_code == 200
        lines = [line for line in stream.text.strip().splitlines() if line.strip()]
        assert any('"stage": "reconcile"' in line for line in lines)
        assert any('"stage": "done"' in line for line in lines)
        # Query file lives outside the scan dir — search must not re-analyze catalog.
        assert faces.analyze_calls == []

    def test_text_search_after_clip_index(self, api_env) -> None:
        client = api_env["client"]
        photo_dir = api_env["dir"]
        store = api_env["store"]
        clip = api_env["clip"]
        path = _save_rgb(photo_dir, "scene.jpg")
        client.put(
            "/settings/scan",
            json={
                "include_directories": [str(photo_dir)],
                "ignore_globs": [],
                "background_indexer_enabled": False,
                "schedule_interval_days": 7,
                "background_modules": ["clip"],
            },
        )
        db = Database(path=api_env["db_path"], vector_store=store)
        with db:
            record = register_file(db, path)
            from indexing.clip import index_clip_image

            index_clip_image(db, store, path, clip, model_version=clip.model_name)
            assert record.id in store.context
        db.close()

        clip.encode_image_calls.clear()
        response = client.post("/search", json={"query": "sunset", "k": 3})
        assert response.status_code == 200
        assert clip.encode_image_calls == []
        assert "matches" in response.json()

    def test_class_search(self, api_env) -> None:
        client = api_env["client"]
        photo_dir = api_env["dir"]
        yolo = api_env["yolo"]
        path = _save_rgb(photo_dir, "dog.jpg")
        yolo.set_detections(
            path,
            [Detection(label="dog", confidence=0.93, bbox=(0, 0, 5, 5))],
        )
        client.put(
            "/settings/scan",
            json={
                "include_directories": [str(photo_dir)],
                "ignore_globs": [],
                "background_indexer_enabled": False,
                "schedule_interval_days": 7,
                "background_modules": ["yolo"],
            },
        )
        response = client.post("/search/class", json={"label": "dog", "k": 5})
        assert response.status_code == 200
        matches = response.json()["matches"]
        assert matches
        assert matches[0]["confidence"] == 0.93
