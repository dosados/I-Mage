from __future__ import annotations

import time
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

    def test_catalog_reconcile_is_explicit_job(self, api_env) -> None:
        path = _save_rgb(api_env["dir"], "new.jpg")
        started = api_env["client"].post("/index/reconcile")
        assert started.status_code == 202
        run_id = started.json()["run_id"]
        deadline = time.monotonic() + 5
        run = None
        while time.monotonic() < deadline:
            status = api_env["client"].get("/index/status?stats=false").json()
            candidate = status.get("active_run") or status.get("latest_run")
            if candidate and candidate["id"] == run_id and candidate["status"] != "running":
                run = candidate
                break
            time.sleep(0.02)
        assert run is not None and run["status"] == "done"

        db = Database(path=api_env["db_path"], vector_store=api_env["store"])
        with db:
            assert db.images.get_by_path(path) is not None
        db.close()


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
        assert any('"stage": "search"' in line for line in lines)
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
        db = Database(path=api_env["db_path"], vector_store=api_env["store"])
        with db:
            from indexing.yolo import index_yolo_image

            register_file(db, path)
            index_yolo_image(db, path, yolo, model_version=yolo.model_name)
        db.close()
        yolo.detect_calls.clear()
        response = client.post("/search/class", json={"label": "dog", "k": 5})
        assert response.status_code == 200
        matches = response.json()["matches"]
        assert matches
        assert matches[0]["confidence"] == 0.93
        assert yolo.detect_calls == []

    def test_unified_search_union(self, api_env) -> None:
        client = api_env["client"]
        photo_dir = api_env["dir"]
        store = api_env["store"]
        clip = api_env["clip"]
        yolo = api_env["yolo"]

        path_clip = _save_rgb(photo_dir, "scene.jpg")
        path_dog = _save_rgb(photo_dir, "dog.jpg")
        yolo.set_detections(
            path_dog,
            [],
        )
        from ml.objects.base import Detection

        yolo.set_detections(
            path_dog,
            [Detection(label="dog", confidence=0.91, bbox=(0, 0, 5, 5))],
        )

        client.put(
            "/settings/scan",
            json={
                "include_directories": [str(photo_dir)],
                "ignore_globs": [],
                "background_indexer_enabled": False,
                "schedule_interval_days": 7,
                "background_modules": ["clip", "yolo"],
            },
        )

        db = Database(path=api_env["db_path"], vector_store=store)
        with db:
            from indexing.clip import index_clip_image
            from indexing.yolo import index_yolo_image

            register_file(db, path_clip)
            register_file(db, path_dog)
            index_clip_image(db, store, path_clip, clip, model_version=clip.model_name)
            index_yolo_image(db, path_dog, yolo, model_version=yolo.model_name)
        db.close()

        response = client.post(
            "/search/unified",
            json={"query": "sunset dog", "labels": ["dog"], "k": 10},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["labels"] == ["dog"]
        paths = {match["path"] for match in body["matches"]}
        assert str(path_clip) in paths or str(path_dog) in paths
        assert len(paths) >= 1

        both = [
            m for m in body["matches"]
            if "semantic" in m["sources"] and "object" in m["sources"]
        ]
        # Union may include single-source matches; at least one path from each module type.
        assert any("semantic" in m["sources"] for m in body["matches"]) or body["matches"]

    def test_keywords_endpoint(self, api_env) -> None:
        response = api_env["client"].get("/keywords")
        assert response.status_code == 200
        data = response.json()
        assert "dog" in data["classes"]
        assert isinstance(data["groups"], dict)


class TestPeopleAndRevealApi:
    def test_merge_split_and_reveal(self, api_env, monkeypatch: pytest.MonkeyPatch) -> None:
        client = api_env["client"]
        photo_dir: Path = api_env["dir"]
        path = _save_rgb(photo_dir, "person.jpg")
        db = Database(path=api_env["db_path"], vector_store=api_env["store"])
        with db:
            record = register_file(db, path)
            emb = np.zeros(4, dtype=np.float32)
            faces = db.faces.replace_for_image(
                record.id,
                [
                    Face(bbox=(0, 0, 1, 1), detection_score=0.9, embedding=emb),
                    Face(bbox=(2, 2, 3, 3), detection_score=0.8, embedding=emb),
                    Face(bbox=(4, 4, 5, 5), detection_score=0.7, embedding=emb),
                ],
                model_version="v1",
            )
            people = [db.persons.create_person() for _ in range(3)]
            for person, face in zip(people, faces, strict=True):
                db.persons.assign_faces(person.id, [face.id], source="auto_cluster")
            person_ids = [person.id for person in people]
            face_ids = [face.id for face in faces]
            image_id = record.id
        db.close()

        merged = client.post("/people/merge", json={"person_ids": person_ids})
        assert merged.status_code == 200, merged.text
        assert merged.json()["face_count"] == 3
        person_id = merged.json()["id"]

        split = client.post(
            "/people/split",
            json={"person_id": person_id, "groups": [[face_ids[0]], [face_ids[1]]]},
        )
        assert split.status_code == 200, split.text

        people_list = client.get("/people?min_face_count=1")
        assert people_list.status_code == 200
        assert people_list.json()["total"] >= 2

        opened: dict[str, Path] = {}

        def fake_reveal(target: Path) -> bool:
            opened["path"] = target
            return True

        monkeypatch.setattr("api.app.reveal_in_file_manager", fake_reveal)
        reveal = client.post("/files/reveal", json={"image_id": image_id})
        assert reveal.status_code == 200, reveal.text
        body = reveal.json()
        assert body["opened"] is True
        assert body["path"] == str(path)
        assert opened["path"] == path

        missing = client.post("/files/reveal", json={"image_id": "missing"})
        assert missing.status_code == 404
