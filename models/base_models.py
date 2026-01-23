from abc import ABC, abstractmethod

import numpy as np
from schemas.contours import Contour
from schemas.service_requests import CompletionRequest


class BaseModel(ABC):
    """ Abstract base class for 2D prompted segmentation models. """
    @abstractmethod
    def process_request(self, image, request: CompletionRequest) -> tuple[np.ndarray, np.ndarray]:
        """ Process a prompted segmentation request.
        :param image: The input image to be segmented.
        :param request: The request to be processed.
        :return: A tuple containing a mask and their corresponding quality score.
        """
        pass
