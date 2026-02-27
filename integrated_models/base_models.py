from abc import ABC, abstractmethod

import numpy as np
import torch
from iquana_toolbox.schemas.contours import Contour
from iquana_toolbox.schemas.service_requests import CompletionRequest


class BaseModel(ABC):
    """ Abstract base class for 2D prompted segmentation integrated_models. """
    def __init__(self):
        super().__init__()
        # Handle device selection automatically
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def process_request(self, image, request: CompletionRequest) -> tuple[np.ndarray, np.ndarray]:
        """ Process a prompted segmentation request.
        :param image: The input image to be segmented.
        :param request: The request to be processed.
        :return: A tuple containing a mask and their corresponding quality score.
        """
        pass
