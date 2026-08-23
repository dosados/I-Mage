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

    @abstractmethod
    def detect_labels(self, image: ImageInput) -> list[str]:
        pass
