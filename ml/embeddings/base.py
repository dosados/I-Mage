from abc import ABC, abstractmethod



class EmbeddingModel(ABC):

    @abstractmethod
    def encode_image(self, image):
        pass

    def encode_images(self, images):
        """Encode a batch, with a compatibility fallback for simple models."""
        return [self.encode_image(image) for image in images]

    @abstractmethod
    def encode_text(self, text):
        pass