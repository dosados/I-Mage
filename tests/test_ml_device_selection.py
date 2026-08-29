from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import ml.embeddings.clip as clip_module
import ml.faces.arcfase_model as face_module
import ml.objects.yolo_model as yolo_module


class DummyClipModel:
    def __init__(self) -> None:
        self.moved_to: str | None = None
        self.eval_called = False

    def to(self, device: str):
        self.moved_to = device
        return self

    def eval(self) -> None:
        self.eval_called = True


@pytest.mark.parametrize(
    ("cuda_available", "expected"),
    [(True, "cuda"), (False, "cpu")],
)
def test_clip_auto_device_is_applied_to_model(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    expected: str,
) -> None:
    dummy = DummyClipModel()
    monkeypatch.setattr(
        clip_module.torch.cuda,
        "is_available",
        lambda: cuda_available,
    )
    monkeypatch.setattr(
        clip_module.CLIPEmbeddingModel,
        "_load_pretrained",
        staticmethod(
            lambda cls, _name: dummy
            if cls is clip_module.CLIPModel
            else object()
        ),
    )

    model = clip_module.CLIPEmbeddingModel()
    assert model.device == expected
    assert dummy.moved_to == expected
    assert dummy.eval_called


class DummyFaceAnalysis:
    instances: list["DummyFaceAnalysis"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.prepared: tuple[int, tuple[int, int]] | None = None
        self.instances.append(self)

    def prepare(self, *, ctx_id: int, det_size: tuple[int, int]) -> None:
        self.prepared = (ctx_id, det_size)

    def get(self, image):
        self.images = getattr(self, "images", [])
        self.images.append(image)
        embedding = np.ones(4, dtype=np.float32)
        face = type(
            "Det",
            (),
            {
                "bbox": np.array([0, 0, 10, 10], dtype=np.float32),
                "det_score": 0.9,
                "normed_embedding": embedding,
            },
        )()
        return [face]


def test_arcface_analyze_batch_loads_then_detects(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    DummyFaceAnalysis.instances.clear()
    monkeypatch.setattr(face_module, "FaceAnalysis", DummyFaceAnalysis)
    monkeypatch.setenv("FACES_ANALYZE_WORKERS", "2")
    from PIL import Image

    paths = []
    for index in range(3):
        path = tmp_path / f"{index}.jpg"
        Image.new("RGB", (16, 16), (index, 0, 0)).save(path)
        paths.append(path)

    model = face_module.ArcFaceRecognizer(device="cpu")
    results = model.analyze_batch(paths)
    assert len(results) == 3
    assert all(len(faces) == 1 for faces in results)
    assert len(DummyFaceAnalysis.instances[-1].images) == 3



@pytest.mark.parametrize(
    ("device", "expected_ctx"),
    [
        ("cpu", -1),
        ("cuda", 0),
        ("cuda:0", 0),
        ("cuda:2", 2),
    ],
)
def test_arcface_maps_device_to_onnx_context(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    expected_ctx: int,
) -> None:
    DummyFaceAnalysis.instances.clear()
    monkeypatch.setattr(face_module, "FaceAnalysis", DummyFaceAnalysis)
    model = face_module.ArcFaceRecognizer(device=device)
    assert model.device == device
    assert DummyFaceAnalysis.instances[-1].prepared == (expected_ctx, (640, 640))


@pytest.mark.parametrize("device", ["cuda:x", "mps", ""])
def test_arcface_rejects_unsupported_explicit_device(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    monkeypatch.setattr(face_module, "FaceAnalysis", DummyFaceAnalysis)
    if device == "":
        monkeypatch.setattr(face_module.torch.cuda, "is_available", lambda: False)
        model = face_module.ArcFaceRecognizer(device=device)
        assert model.device == "cpu"
        return
    with pytest.raises(ValueError):
        face_module.ArcFaceRecognizer(device=device)


class DummyYolo:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name


@pytest.mark.parametrize(
    ("cuda_available", "expected"),
    [(True, "cuda:0"), (False, "cpu")],
)
def test_yolo_auto_device_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    expected: str,
) -> None:
    monkeypatch.setattr(yolo_module, "YOLO", DummyYolo)
    monkeypatch.setattr(
        yolo_module.torch.cuda,
        "is_available",
        lambda: cuda_available,
    )
    model = yolo_module.YoloObjectsRetriever(model_name="weights.pt")
    assert model.device == expected


def test_yolo_respects_explicit_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yolo_module, "YOLO", DummyYolo)
    monkeypatch.setattr(yolo_module.torch.cuda, "is_available", lambda: True)
    model = yolo_module.YoloObjectsRetriever(
        model_name="weights.pt",
        device="cpu",
    )
    assert model.device == "cpu"
