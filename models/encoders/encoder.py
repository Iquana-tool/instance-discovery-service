from abc import ABC, abstractmethod
from typing import Literal

from PIL.Image import Image
from torch import Tensor


class Encoder(ABC):
    @abstractmethod
    def embed_image(self,
                    image: Image,
                    standardize: bool = True,
                    reduced_features: int | None = None,
                    reduce_method: Literal["pca"] = "pca",
                    keep_dim: bool = True,
                    debug_pca: bool = False) -> Tensor:
        """ Embed an image into feature vectors.
        :param image: Image to be embedded in feature vectors.
        :param standardize: Whether to standardize the embedding.
        :param reduced_features: Number of features to reduce the embedding to. Must be smaller than the number of
            features produced by the encoder.
        :param reduce_method: Method used to reduce the embedding. Eg. PCA.
        :param keep_dim: Keep original dimensions. If false, might resize the image.
        :param debug_pca: If true, plots a PCA plot of the embedding. Can be used for debugging the model and checking
            whether everything is ok.
        """
        pass

