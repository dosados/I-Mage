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

        ctx_id = 0 if self.device == "cuda" else -1
        self._app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection", "recognition"],
        )
        self._app.prepare(ctx_id=ctx_id, det_size=det_size)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def analyze(self, image: ImageInput) -> list[Face]:
        detected_faces = self._app.get(self._load_image(image))

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
