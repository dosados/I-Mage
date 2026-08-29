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

    def analyze_batch(
        self,
        images: list[ImageInput],
        *,
        should_stop=None,
    ) -> list[list[Face]]:
        """Analyze a batch; fakes fall back to serial ``analyze``."""
        results: list[list[Face]] = []
        for image in images:
            if should_stop is not None and should_stop():
                raise InterruptedError("face analysis stopped")
            results.append(self.analyze(image))
        return results
