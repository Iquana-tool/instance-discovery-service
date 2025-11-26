from abc import ABC, abstractmethod

from PIL.Image import Image
from torch import Tensor


class Encoder(ABC):
    @abstractmethod
    def embed_image(self, image: Image, keep_dim=True) -> Tensor:
        """ Embed an image into feature vectors.
        :param image: Image to be embedded in feature vectors.
        :param keep_dim: Keep original dimensions. If false, might resize the image.
        """
        pass

