from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ml.objects.base import ImageInput


@dataclass(frozen=True)
class Face:
    bbox: tuple[float, float, float, float]
    detection_score: float
    embedding: np.ndarray


class FaceRecognizer(ABC):
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """For DB schema / index validation."""

    @abstractmethod
    def analyze(self, image: ImageInput) -> list[Face]:
        pass
