import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms


def iom(mask1, mask2):
    intersection = np.sum(np.logical_and(mask1, mask2)).item()
    minimum = min(np.sum(mask1).item(), np.sum(mask2).item())
    return intersection / minimum


def filter_seed_masks(
        combined_seed_mask: np.ndarray,
        new_masks: list[np.ndarray],
        iom_threshold: float = 0.) -> list:
    keep_idx: list[int] = []
    for j, new_mask in enumerate(new_masks):
        if iom(combined_seed_mask, new_mask) <= iom_threshold:
            keep_idx.append(j)
    return keep_idx


def filter_seed_bboxes(combined_seed_mask: np.ndarray, bboxes: list[np.ndarray], iom_threshold=0.0) -> list:
    keep_idx = []
    for i, box in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, box)
        # Create a mask for the current box
        box_mask = np.zeros_like(combined_seed_mask, dtype=np.bool)
        box_mask[y1:y2, x1:x2] = True
        # Only keep boxes with minimal overlap (e.g., < 50% of box area) and inside a certain area range
        if iom(combined_seed_mask, box_mask) <= iom_threshold:
            keep_idx.append(i)
    return keep_idx


def extract_masklets(binary_mask: np.ndarray, pixel_scores=None) -> tuple[np.ndarray, np.ndarray]:
    # Connected component analysis for bounding boxes
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    # Extract individual components
    masklets = []
    scores = []
    for i in range(1, num_labels):
        masklet = labels == i
        masklets.append(masklet)
        if pixel_scores is not None:
            scores.append(np.average(pixel_scores[masklet]).item())
        else:
            scores.append(1.)
    return np.array(masklets), np.array(scores)
