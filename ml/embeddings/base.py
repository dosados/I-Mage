from abc import ABC, abstractmethod



class EmbeddingModel(ABC):

    @abstractmethod
    def encode_image(self, image):
        pass

    @abstractmethod
    def encode_text(self, text):
        pass