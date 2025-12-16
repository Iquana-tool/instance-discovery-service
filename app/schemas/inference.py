from typing import Any, Literal, LiteralString

import cv2
import numpy as np
import torch
from PIL.Image import fromarray
from pydantic import BaseModel, Field


class Request(BaseModel):
    model_key: str = Field(..., description="The key of the model.")
    user_id: str = Field(..., description="The user id of the model.")
    seeds: list[list[list[bool]]] = Field(...,
                                                description="Seeds is a list of binary masks.")

    def get_combined_seed_mask(self, size) -> np.ndarray:
        combined_seed_mask = np.zeros(self.max_image_size, dtype=np.bool)
        min_area, max_area = 1, 0
        for seed in self.seeds:
            seed_mask = np.array(seed, dtype=np.bool)
            min_area = min(min_area, np.count_nonzero(seed_mask) / seed_mask.size)
            max_area = max(max_area, np.count_nonzero(seed_mask) / seed_mask.size)
            seed_mask = np.array(fromarray(seed_mask).resize(size))
            combined_seed_mask = np.logical_or(combined_seed_mask, seed_mask)
        return combined_seed_mask

    @property
    def min_max_area(self) -> tuple[float, float]:
        min_area, max_area = 1., 0.
        for seed in self.seeds:
            seed_mask = np.array(seed, dtype=np.bool)
            min_area = min(min_area, np.count_nonzero(seed_mask) / seed_mask.size)
            max_area = max(max_area, np.count_nonzero(seed_mask) / seed_mask.size)
        return min_area, max_area

    def get_bboxes(self,
                   format: Literal["xywh", "x1y1x2y2"] = "x1y1x2y2",
                   return_tensors: bool = True,
                   device: Literal["cpu", "cuda"] | None = None,
                   relative_coordinates: bool = True,
                   resize_to: None | tuple[int, int] = None) \
            -> list[list[float]] | torch.Tensor:
        bboxes = []
        for seed in self.seeds:
            seed_mask = np.array(seed, dtype=np.bool)
            if resize_to:
                seed_mask = cv2.resize(seed_mask.astype(np.uint8), resize_to)
            indices = np.argwhere(seed_mask.astype(bool))
            x_min = np.min(indices[1]).item()
            y_min = np.min(indices[0]).item()
            x_max = np.max(indices[1]).item()
            y_max = np.max(indices[0]).item()
            if relative_coordinates:
                x_min /= seed_mask.shape[1]
                y_min /= seed_mask.shape[0]
                x_max /= seed_mask.shape[1]
                y_max /= seed_mask.shape[0]
            if format == "xywh":
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            elif format == "x1y1x2y2":
                bbox = [x_min, y_min, x_max, y_max]
            else:
                raise ValueError("Format must be either 'xywh' or 'x1y1x2y2'.")
            bboxes.append(bbox)
        if return_tensors:
            return torch.tensor(bboxes).float().to(device)
        else:
            return bboxes


class InstanceMasksResponse(BaseModel):
    type: str = "instance_masks"
    masks: list[list[list[bool]]] = Field(..., description="Masks is a list of binary masks. One for each object.")
    scores: list[float] = Field(..., description="Scores is a list of float values indicating the confidence of each mask.")

    @property
    def n_objects(self) -> int:
        return len(self.masks)


class BBoxesResponse(BaseModel):
    type: str = "bboxes"
    bboxes: list[list[float]] = Field(..., description="Bounding boxes is a list of float value indicating the coordinates of each bounding box.")
    scores: list[float] = Field(..., description="Scores is a list of float value indicating the confidence of each bounding box.")

    @property
    def n_objects(self) -> int:
        return len(self.bboxes)
