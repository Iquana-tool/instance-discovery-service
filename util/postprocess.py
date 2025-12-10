import numpy as np

def iom(mask1, mask2):
    intersection = np.sum(np.logical_and(mask1, mask2)).item()
    minimum = min(np.sum(mask1).item(), np.sum(mask2).item())
    return intersection / minimum


def filter_seed_masks(combined_seed_mask: np.ndarray, new_masks: list[np.ndarray]) -> list:
    keep_idx: list[int] = []
    for j, new_mask in enumerate(new_masks):
        if iom(combined_seed_mask, new_mask) < 0.2:
            keep_idx.append(j)
    return keep_idx


def filter_seed_bboxes(combined_seed_mask: np.ndarray, bboxes: list[np.ndarray]) -> list:
    keep_idx = []
    for i, box in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, box)
        # Create a mask for the current box
        box_mask = np.zeros_like(combined_seed_mask, dtype=np.bool)
        box_mask[y1:y2, x1:x2] = True
        # Only keep boxes with minimal overlap (e.g., < 50% of box area) and inside a certain area range
        if iom(combined_seed_mask, box_mask) < 0.5:
            keep_idx.append(i)
    return keep_idx
