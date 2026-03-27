from abc import ABC, abstractmethod

import numpy as np
import torch
from iquana_toolbox.schemas.networking.http.services import CompletionRequest


class BaseModel(torch.nn.Module, ABC):
    """ Abstract base class for 2D prompted segmentation models. """
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
