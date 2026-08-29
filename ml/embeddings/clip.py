from pathlib import Path

import logging

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from ml.embeddings.base import EmbeddingModel

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"

ImageInput = str | Path | Image.Image


class CLIPEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Prefer the local HF cache. A slow/hung Hub check on startup is what
        # made uvicorn appear frozen (and ignore Ctrl+C) for minutes.
        self.model = self._load_pretrained(CLIPModel, model_name).to(self.device)
        self.processor = self._load_pretrained(CLIPProcessor, model_name)
        self.model.eval()

    @staticmethod
    def _load_pretrained(cls, model_name: str):
        try:
            return cls.from_pretrained(model_name, local_files_only=True)
        except Exception:
            logger.warning(
                "local cache miss for %s (%s); downloading from Hugging Face…",
                model_name,
                cls.__name__,
            )
            return cls.from_pretrained(model_name)

    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)

        return self._normalize(self._extract_features(outputs))

    def encode_image(self, image: ImageInput) -> np.ndarray:
        return self.encode_images([image])[0]

    def encode_images(self, images: list[ImageInput]) -> list[np.ndarray]:
        if not images:
            return []
        pil_images = [self._load_image(image) for image in images]
        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)

        features = self._extract_features(outputs)
        normalized = features / features.norm(dim=-1, keepdim=True)
        array = normalized.cpu().numpy().astype(np.float32)
        return [row for row in array]

    def _load_image(self, image: ImageInput) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.open(image).convert("RGB")

    def _extract_features(self, outputs: torch.Tensor | object) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return outputs
        return outputs.pooler_output

    def _normalize(self, features: torch.Tensor) -> np.ndarray:
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().squeeze(0).astype(np.float32)
