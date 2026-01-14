from typing import Union

import cv2
import numpy
import numpy as np
import torch
from PIL.Image import Image, fromarray
from app.schemas.inference import BBoxesResponse, InstanceMasksResponse
from models.base_models import BaseModel
from models.encoders.dino_encoder import DinoModel, DinoModelType
from models.encoders.encoder_base_class import Encoder
from models.similarity.cosine_similarity import CosineSimilarity
from util.postprocess import extract_masklets


class SimpleThresholdingModel(BaseModel):
    def __init__(self,
                 similarity: CosineSimilarity,
                 backbone: Encoder,
                 max_image_size: Union[int, list[int]] = 512):
        self.similarity = similarity
        self.backbone = backbone
        if type(max_image_size) == int:
            self.max_image_size = [max_image_size, max_image_size]
        else:
            self.max_image_size = max_image_size

    def process_request(self, image, request):
        # Preprocess image
        if isinstance(image, np.ndarray):
            image = fromarray(image)

        # Resize to a fixed size to control computational efforts as images can get large
        image = image.resize(self.max_image_size)

        # Embeds the image and standardizes embeddings
        embedded_img = self.backbone.embed_image(
            image=image,
            standardize=True
        )

        # Combine all seed masks into a single binary mask
        combined_seed_mask = request.get_combined_seed_mask(self.max_image_size)

        # Process seeds separately and average similarity maps
        sim_maps = []
        for mask in request.positive_masks:
            seed_mask = np.array(mask, dtype=np.bool)
            seed_mask = np.array(fromarray(seed_mask).resize(self.max_image_size))
            self.similarity.reset()
            self.similarity.add_seed_instance(embedded_img[seed_mask])
            sim_map = self.similarity.get_similarity_map(embedded_img)
            sim_maps.append(sim_map)
        final_sim_map = torch.mean(torch.stack(sim_maps), dim=0).cpu().numpy()
        final_sim_map = (final_sim_map * 255).astype(np.uint8)

        # Adaptive thresholding
        # Takes the minimum required to recreate the seed masks
        threshold = np.median(final_sim_map[combined_seed_mask]).item()
        print(f"Threshold: {threshold}")
        _, thresholded = cv2.threshold(final_sim_map, threshold, 255, cv2.THRESH_BINARY)

        masklets, scores = extract_masklets(thresholded, final_sim_map)

        return InstanceMasksResponse.from_masks(masklets, scores, request)


class Dino1000CosineHeMaxAgg(SimpleThresholdingModel):
    def __init__(self):
        super().__init__(
            max_image_size=512,
            similarity=CosineSimilarity(
                device="auto",
                memory_aggregation="none",
                similarity_aggregation="mean",
                similarity_redistribution_method="he"),
            backbone=DinoModel(
                device="auto",
                model_type=DinoModelType.VITL16,
                patch_size=16,
                image_size=1024,
            )
        )
