from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = [
    pytest.mark.ml,
    pytest.mark.cuda,
    pytest.mark.slow,
]


def _require_real_ml() -> None:
    if os.environ.get("RUN_ML_TESTS") != "1":
        pytest.skip("set RUN_ML_TESTS=1 to load real model weights")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


@pytest.fixture(scope="module")
def sample_image() -> Image.Image:
    try:
        from skimage import data
    except ImportError:
        pytest.skip("scikit-image is required for the bundled astronaut image")
    return Image.fromarray(data.astronaut()).convert("RGB")


@pytest.fixture(scope="module")
def clip_model():
    _require_real_ml()
    from ml.embeddings.clip import CLIPEmbeddingModel

    return CLIPEmbeddingModel()


@pytest.fixture(scope="module")
def yolo_model():
    _require_real_ml()
    from ml.objects.yolo_model import DEFAULT_MODEL_PATH, YoloObjectsRetriever

    assert Path(DEFAULT_MODEL_PATH).is_file(), (
        f"YOLO weights are missing at {DEFAULT_MODEL_PATH}; "
        "place yolov8s.pt in artifacts/"
    )
    return YoloObjectsRetriever()


@pytest.fixture(scope="module")
def face_model():
    _require_real_ml()
    from ml.faces.arcfase_model import ArcFaceRecognizer

    return ArcFaceRecognizer()


def test_clip_loads_on_cuda_and_returns_normalized_embeddings(
    clip_model, sample_image: Image.Image
) -> None:
    parameter_device = next(clip_model.model.parameters()).device
    assert clip_model.device == "cuda"
    assert parameter_device.type == "cuda"

    text = clip_model.encode_text("a portrait photo of a person")
    image = clip_model.encode_image(sample_image)
    assert text.shape == image.shape == (512,)
    assert text.dtype == image.dtype == np.float32
    assert np.linalg.norm(text) == pytest.approx(1.0, abs=1e-5)
    assert np.linalg.norm(image) == pytest.approx(1.0, abs=1e-5)
    assert np.isfinite(text).all() and np.isfinite(image).all()


def test_yolo_loads_on_cuda_and_detects_a_person(
    yolo_model, sample_image: Image.Image
) -> None:
    detections = yolo_model.detect(sample_image)
    model_device = next(yolo_model.model.model.parameters()).device
    assert model_device.type == "cuda"
    assert detections
    assert any(item.label == "person" for item in detections)
    assert all(0.0 <= item.confidence <= 1.0 for item in detections)
    assert detections == sorted(
        detections, key=lambda item: item.confidence, reverse=True
    )


def test_arcface_uses_cuda_provider_and_returns_normalized_face(
    face_model, sample_image: Image.Image
) -> None:
    providers: set[str] = set()
    for component in face_model._app.models.values():
        session = getattr(component, "session", None)
        if session is not None:
            providers.update(session.get_providers())

    assert face_model.device == "cuda"
    assert "CUDAExecutionProvider" in providers, (
        "ArcFace requested CUDA but ONNX Runtime did not activate "
        f"CUDAExecutionProvider; active providers: {sorted(providers)}"
    )

    faces = face_model.analyze(sample_image)
    assert faces
    assert all(face.embedding.shape == (512,) for face in faces)
    assert all(face.embedding.dtype == np.float32 for face in faces)
    assert all(np.linalg.norm(face.embedding) == pytest.approx(1.0, abs=1e-5) for face in faces)
    assert faces == sorted(faces, key=lambda item: item.detection_score, reverse=True)


def test_models_allocate_cuda_memory_and_synchronize(
    clip_model, yolo_model, face_model, sample_image: Image.Image
) -> None:
    torch.cuda.reset_peak_memory_stats()
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    started.record()
    for _ in range(4):
        clip_model.encode_image(sample_image)
        yolo_model.detect(sample_image)
        face_model.analyze(sample_image)
    finished.record()
    torch.cuda.synchronize()

    assert torch.cuda.max_memory_allocated() > 0
    assert started.elapsed_time(finished) > 0
