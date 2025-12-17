from typing import Union

import cv2
import numpy as np
import torch
from PIL.Image import Image, fromarray
from torchvision.ops import batched_nms

from app.schemas.inference import BBoxesResponse
from models.base_models import BaseModel
from models.encoders.dino import DinoModel, DinoModelType
from models.encoders.encoder import Encoder
from models.predictors.cosine_similarity_predictor import CosineSimilarityPredictor
from util.postprocess import filter_seed_bboxes


class CosineSimilarityModel(BaseModel):
    def __init__(self,
                 predictor: CosineSimilarityPredictor,
                 backbone: Encoder,
                 max_image_size: Union[int, list[int]] = 512):
        self.predictor = predictor
        self.backbone = backbone
        if type(max_image_size) == int:
            self.max_image_size = [max_image_size, max_image_size]
        else:
            self.max_image_size = max_image_size

    def process_request(self, image, request):
        # 1. Preprocess image
        if isinstance(image, np.ndarray):
            image = fromarray(image)
        image = image.resize(self.max_image_size)
        print("Embedding image!")
        embedded_img = self.backbone.embed_image(image=image)
        print(f"Embedded image shape: {embedded_img.shape}")

        # 2. Combine all seed masks into a single binary mask
        combined_seed_mask = request.get_combined_seed_mask(self.max_image_size)
        min_area, max_area = request.min_max_area
        min_area = max(0., min_area * 0.8)
        max_area = min(1., max_area * 1.2)

        # 3. Process seeds separately and average similarity maps
        sim_maps = []
        for seed in request.seeds:
            seed_mask = np.array(seed, dtype=np.bool)
            seed_mask = np.array(fromarray(seed_mask).resize(self.max_image_size))
            self.predictor.reset()
            self.predictor.add_seed_instance(embedded_img[seed_mask])
            sim_map = self.predictor.get_similarity_map(embedded_img)
            sim_maps.append(sim_map)
        final_sim_map = torch.mean(torch.stack(sim_maps), dim=0).cpu().numpy()
        final_sim_map = (final_sim_map * 255).astype(np.uint8)
        print("Computed averaged similarity map.")

        # 4. Adaptive thresholding
        _, thresholded = cv2.threshold(final_sim_map, 230, 255, cv2.THRESH_BINARY)
        print("Applied adaptive thresholding.")

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


class Dino1000CosineHeMaxAgg(CosineSimilarityModel):
    def __init__(self):
        super().__init__(
            max_image_size=512,
            predictor=CosineSimilarityPredictor(
                device="auto",
                memory_aggregation="none",
                similarity_aggregation="mean",
                similarity_redistribution_method="norm"),
            backbone=DinoModel(
                device="auto",
                model_type=DinoModelType.VITL16,
                patch_size=16,
                image_size=1024,
            )
        )
