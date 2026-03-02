from abc import ABC, abstractmethod
import torch
import numpy as np


class SimilarityMetric(ABC):
    @abstractmethod
    def reset(self):
        """ Reset the predictor to its initial state. """
        pass

    @abstractmethod
    def add_seed_instance(self, seed_instance):
        """ Add an instance as a seed for prediction. """
        pass

    def add_seed_instances(self, seed_instances):
        for seed_instance in seed_instances:
            self.add_seed_instance(seed_instance)

    @abstractmethod
    def predict(self, encoding: torch.tensor) -> torch.tensor:
        pass
