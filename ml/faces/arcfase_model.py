import concurrent.futures as cf
import os

import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis
from PIL import Image

from ml.faces.base import Face, FaceRecognizer
from ml.objects.base import ImageInput

DEFAULT_MODEL_NAME = "buffalo_l"
DEFAULT_EMBEDDING_DIM = 512
DEFAULT_DET_SIZE = (640, 640)


class ArcFaceRecognizer(FaceRecognizer):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        det_size: tuple[int, int] = DEFAULT_DET_SIZE,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.det_size = det_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._embedding_dim = DEFAULT_EMBEDDING_DIM

        if self.device == "cpu":
            ctx_id = -1
        elif self.device == "cuda":
            ctx_id = 0
        elif self.device.startswith("cuda:"):
            try:
                ctx_id = int(self.device.split(":", maxsplit=1)[1])
            except ValueError as exc:
                raise ValueError(f"invalid CUDA device: {self.device}") from exc
        else:
            raise ValueError(f"unsupported device: {self.device}")
        self._app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection", "recognition"],
        )
        self._app.prepare(ctx_id=ctx_id, det_size=det_size)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def analyze(self, image: ImageInput) -> list[Face]:
        return self._from_bgr(self._load_image(image))

    def analyze_batch(
        self,
        images: list[ImageInput],
        *,
        should_stop=None,
    ) -> list[list[Face]]:
        if not images:
            return []
        loaded = self._load_images(images, should_stop=should_stop)
        results: list[list[Face]] = []
        for array in loaded:
            if should_stop is not None and should_stop():
                raise InterruptedError("face analysis stopped")
            results.append(self._from_bgr(array))
        return results

    def _load_images(
        self,
        images: list[ImageInput],
        *,
        should_stop=None,
    ) -> list[np.ndarray]:
        workers = max(1, int(os.environ.get("FACES_ANALYZE_WORKERS", "4")))
        if workers == 1 or len(images) == 1:
            arrays = []
            for image in images:
                if should_stop is not None and should_stop():
                    raise InterruptedError("face analysis stopped")
                arrays.append(self._load_image(image))
            return arrays

        with cf.ThreadPoolExecutor(max_workers=min(workers, len(images))) as pool:
            futures = [pool.submit(self._load_image, image) for image in images]
            arrays: list[np.ndarray] = []
            for future in futures:
                if should_stop is not None and should_stop():
                    for pending in futures:
                        pending.cancel()
                    raise InterruptedError("face analysis stopped")
                arrays.append(future.result())
            return arrays

    def _from_bgr(self, array: np.ndarray) -> list[Face]:
        detected_faces = self._app.get(array)
        faces = [
            Face(
                bbox=tuple(float(value) for value in face.bbox.tolist()),
                detection_score=float(face.det_score),
                embedding=self._normalize_embedding(face.normed_embedding),
            )
            for face in detected_faces
        ]
        faces.sort(key=lambda item: item.detection_score, reverse=True)
        return faces

    def _load_image(self, image: ImageInput) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image
        if isinstance(image, Image.Image):
            rgb = np.array(image.convert("RGB"))
        else:
            rgb = np.array(Image.open(image).convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.astype(np.float32)
