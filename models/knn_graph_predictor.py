from typing import Union

import cv2
import torch
import numpy as np
from PIL.Image import fromarray
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components
from scipy.sparse import lil_matrix
from torchvision.ops import batched_nms

from app.schemas.inference import BBoxesResponse
from models.base_models import BaseModel
from models.encoders.dino import DinoModelType, DinoModel
from models.encoders.encoder import Encoder
from util.postprocess import filter_seed_bboxes


def graph_based_propagation(embedded_img, seed_mask, k=5, sigma=1.0):
    """
    Propagate labels from seed pixels to similar regions using a k-NN graph.

    Args:
        embedded_img: Tensor of shape (H, W, D) containing embeddings.
        seed_mask: Binary mask of shape (H, W) indicating seed pixels.
        k: Number of nearest neighbors for graph construction.
        sigma: Scaling factor for affinity matrix.

    Returns:
        label_mask: Binary mask of shape (H, W) indicating propagated labels.
    """
    H, W, D = embedded_img.shape
    n_pixels = H * W

    # Flatten embeddings and seed mask
    embeddings = embedded_img.reshape(n_pixels, D)
    seeds = seed_mask.reshape(n_pixels)

    # Build k-NN graph
    nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # Construct affinity matrix
    affinity = lil_matrix((n_pixels, n_pixels), dtype=np.float32)
    for i in range(n_pixels):
        for j, dist in zip(indices[i], distances[i]):
            if i != j:
                affinity[i, j] = np.exp(-dist**2 / (2 * sigma**2))

    # Propagate labels
    labels = -1 * np.ones(n_pixels, dtype=np.int32)
    labels[seeds] = 1  # Label seeds as 1

    # Diffuse labels through the graph
    for _ in range(3):  # Number of diffusion steps
        new_labels = labels.copy()
        for i in range(n_pixels):
            if labels[i] == 1:  # Skip already labeled seeds
                continue
            # Get neighbors
            neighbors = affinity.rows[i]
            neighbor_labels = labels[neighbors]
            # If any neighbor is labeled, propagate the label
            if np.any(neighbor_labels == 1):
                new_labels[i] = 1
        labels = new_labels

    # Reshape labels to mask
    label_mask = labels.reshape(H, W)
    return label_mask.astype(bool)


class KNNGraphPredictor(BaseModel):
    def __init__(self,
                 k,
                 sigma,
                 backbone: Encoder = None,
                 max_image_size: Union[int, list[int]] = 132):
        self.k = k
        self.sigma = sigma
        if type(max_image_size) == int:
            self.max_image_size = [max_image_size, max_image_size]
        else:
            self.max_image_size = max_image_size
        if backbone is None:
            self.backbone = DinoModel(
                                        device="auto",
                                        model_type=DinoModelType.VITL16,
                                        patch_size=16,
                                        image_size=512,
                                    )
        else:
            self.backbone = backbone

    def process_request(self, image, request):
        # 1. Preprocess image
        if isinstance(image, np.ndarray):
            image = fromarray(image)
        image = image.resize(self.max_image_size)
        print("Embedding image!")
        embedded_img = self.backbone.embed_image(image=image)
        print(f"Embedded image shape: {embedded_img.shape}")

        # Combine all seed masks into a single binary mask
        combined_seed_mask = request.get_combined_seed_mask(self.max_image_size)

        # Graph-based propagation
        propagated_mask = graph_based_propagation(embedded_img.cpu().numpy(), combined_seed_mask, self.k, self.sigma)

        # Use propagated_mask for further processing (e.g., bounding boxes)
        # For example, replace `final_sim_map` with `propagated_mask` for thresholding
        _, thresholded = cv2.threshold(propagated_mask.astype(np.uint8) * 255, 127, 255, cv2.THRESH_BINARY)

        # 5. Connected component analysis for bounding boxes
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresholded, connectivity=8)
        boxes = []
        for i in range(1, num_labels):
            x, y, w, h, _ = stats[i]
            boxes.append([x, y, x + w, y + h])  # [x1, y1, x2, y2] format

        # 6. Non-Maximum Suppression (NMS)
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            keep = batched_nms(boxes, areas, torch.zeros(len(boxes)), iou_threshold=0.5)
            final_boxes = boxes[keep].numpy()
        else:
            final_boxes = np.array([])

        # 7. Remove boxes overlapping with seeds
        keep_idx = filter_seed_bboxes(combined_seed_mask, final_boxes)
        filtered_boxes = final_boxes[keep_idx]

        # 8. Normalize box coordinates
        h, w = final_sim_map.shape
        normalized_boxes = []
        scores = []
        for box in filtered_boxes:
            x1, y1, x2, y2 = box.astype(int)
            scores.append(np.average(final_sim_map[y1:y2, x1:x2]).item())
            norm_x1 = float(x1 / w)
            norm_y1 = float(y1 / h)
            norm_x2 = float(x2 / w)
            norm_y2 = float(y2 / h)
            normalized_boxes.append([
                norm_x1, norm_y1, norm_x2, norm_y2,
            ])

        print(f"Detected {len(normalized_boxes)} objects after filtering seed overlaps.")
        return BBoxesResponse(
            bboxes=normalized_boxes,
            scores=scores
        )
