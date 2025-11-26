from abc import ABC, abstractmethod


class Encoder(ABC):
    @abstractmethod
    def embed_image(self, image, keep_dim=True):
        """ Embed an image into feature vectors.
        :param image: Image to be embedded in feature vectors.
        :param keep_dim: Keep original dimensions. If false, might resize the image.
        """
        pass

