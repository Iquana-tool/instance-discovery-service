from pycocotools import mask as maskUtils
import numpy as np
from typing import Literal, Any

import cv2
import numpy as np
import torch
from pydantic import BaseModel, Field

from util.postprocess import filter_seed_masks


class Request(BaseModel):
    model_key: str = Field(..., description="The key of the model.")
    user_id: str = Field(..., description="The user id of the model.")
    seeds: list[dict] = Field(..., description="Seeds is a list of rle encoded binary masks")
    negative_seeds: list[dict] | None = Field(default=[], description="Negative seeds is a list of rle encoded binary masks")
    concept: str | None = Field(default=None,
                                description="Optional str describing the concept of the objects to be detected.")

    @property
    def positive_masks(self) -> list[np.ndarray]:
        """ Returns an list of binary masks. Internally decodes the rle encoded masks."""
        masks = []
        for rle in self.seeds:
            masks.append(maskUtils.decode(rle))
        return masks

    @property
    def negative_masks(self):
        """ Returns an list of binary masks. Internally decodes the rle encoded masks."""
        masks = []
        for rle in self.negative_seeds:
            masks.append(maskUtils.decode(rle))
        return masks

    def get_combined_seed_mask(self, size) -> np.ndarray:
        """ Returns all seed masks as one binary mask of the given size. """
        combined_seed_mask = np.zeros(tuple(size), dtype=bool)
        for mask in self.positive_masks:
            seed_mask = np.array(mask, dtype=bool)
            seed_mask = cv2.resize(seed_mask.astype(np.uint8), size[::-1]).astype(bool)
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
                   format: Literal["xywh", "x1y1x2y2", "cxcywh"] = "x1y1x2y2",
                   return_tensors: bool = True,
                   device: Literal["cpu", "cuda"] | None = None,
                   relative_coordinates: bool = True,
                   resize_to: None | tuple[int, int] = None) \
            -> list[list[float]] | torch.Tensor:
        bboxes = []
        for mask in self.positive_masks:
            seed_mask = np.array(mask, dtype=np.bool)
            if resize_to:
                seed_mask = cv2.resize(seed_mask.astype(np.uint8), resize_to)
            indices = np.argwhere(seed_mask.astype(bool))
            x_min = np.min(indices[0]).item()
            y_min = np.min(indices[1]).item()
            x_max = np.max(indices[0]).item()
            y_max = np.max(indices[1]).item()
            if relative_coordinates:
                x_min /= seed_mask.shape[1]
                y_min /= seed_mask.shape[0]
                x_max /= seed_mask.shape[1]
                y_max /= seed_mask.shape[0]
            if format == "xywh":
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            elif format == "x1y1x2y2":
                bbox = [x_min, y_min, x_max, y_max]
            elif format == "cxcywh":
                w = x_max - x_min
                h = y_max - y_min
                cx = x_min + w / 2
                cy = y_min + h / 2
                bbox = [cx, cy, w, h]
            else:
                raise ValueError("Format must be either 'xywh' or 'x1y1x2y2'.")
            bboxes.append(bbox)
        if return_tensors:
            return torch.tensor(bboxes).float().to(device)
        else:
            return bboxes


class InstanceMasksResponse(BaseModel):
    type: str = "instance_masks"
    masks: list[dict] = Field(..., description="Masks is a list of rle encoded masks. One for each object.")
    scores: list[float] = Field(..., description="Scores is a list of float values indicating the confidence of each mask.")

    @property
    def n_objects(self) -> int:
        return len(self.masks)

    @classmethod
    def from_masks(cls, masks: np.ndarray, scores, postprocess_request: Request | None = None):
        if postprocess_request is not None:
            # We postprocess the results with our request, eg by filtering masks that were already in the request.
            keep_ids = filter_seed_masks(postprocess_request.get_combined_seed_mask(masks.shape[1:]),
                                         masks)
            if len(keep_ids) < len(masks):
                print(f"After filtering with seeds: {len(keep_ids)}")
            masks = masks[keep_ids]
        rle_masks = []
        for mask in masks:
            rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
            rle_masks.append(rle)
        if isinstance(scores, np.ndarray):
            scores = scores.tolist()
        return cls(
            masks=rle_masks,
            scores=scores,
        )


class BBoxesResponse(BaseModel):
    type: str = "bboxes"
    bboxes: list[list[float]] = Field(..., description="Bounding boxes is a list of float value indicating the coordinates of each bounding box.")
    scores: list[float] = Field(..., description="Scores is a list of float value indicating the confidence of each bounding box.")

    @property
    def n_objects(self) -> int:
        return len(self.bboxes)
