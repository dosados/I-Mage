from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

ImageInput = str | Path | Image.Image


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]


class ObjectsRetriever(ABC):
    @abstractmethod
    def detect(self, image: ImageInput) -> list[Detection]:
        pass

    def detect_batch(self, images: list[ImageInput]) -> list[list[Detection]]:
        """Detect objects in a batch, falling back to serial calls for fakes."""
        return [self.detect(image) for image in images]

    @abstractmethod
    def detect_labels(self, image: ImageInput) -> list[str]:
        pass
