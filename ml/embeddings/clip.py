from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from ml.embeddings.base import EmbeddingModel

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

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

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
        pil_image = self._load_image(image)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)

        return self._normalize(self._extract_features(outputs))

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
