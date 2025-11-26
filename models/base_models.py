import numpy as np

from app.schemas.inference import Request
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """ Abstract base class for 2D prompted segmentation models. """
    @abstractmethod
    def process_request(self, image, request: Request) -> tuple[np.ndarray, float]:
        """ Process a prompted segmentation request.
        :param image: The input image to be segmented.
        :param request: The request to be processed.
        :return: A tuple containing a mask and their corresponding quality score.
        """
        pass
