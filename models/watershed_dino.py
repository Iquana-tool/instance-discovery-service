import cv2
import numpy as np
import torch
from PIL.Image import fromarray
from skimage.segmentation import watershed

from models.base_models import BaseModel
from models.encoders.dino_encoder import DinoModel, DinoModelType
from models.similarity.cosine_similarity import CosineSimilarity


class WatershedDINO(BaseModel):
    def __init__(self,
                 max_image_size=1024,
                 similarity=CosineSimilarity(
                     device="auto",
                     memory_aggregation="none",
                     similarity_aggregation="mean",
                     similarity_redistribution_method="none"),
                 backbone=DinoModel(
                     device="auto",
                     model_type=DinoModelType.VITL16,
                     patch_size=16,
                     image_size=1024,
                 )):
        super().__init__()
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
        combined_seed_mask = request.combined_exemplar_mask.astype(np.uint8)
        combined_seed_mask = cv2.resize(combined_seed_mask, self.max_image_size)

        # Process seeds separately and average similarity maps
        sim_maps = []
        for mask in request.positive_exemplar_masks:
            seed_mask = np.array(fromarray(mask).resize(self.max_image_size)).astype(bool)
            self.similarity.reset()
            self.similarity.add_seed_instance(embedded_img[seed_mask])
            sim_map = self.similarity.get_similarity_map(embedded_img)
            sim_maps.append(sim_map)
        final_sim_map = torch.mean(torch.stack(sim_maps), dim=0).cpu().numpy()
        final_sim_map = (final_sim_map * 255).astype(np.uint8)

        # Watershed Algorithm
        masklets = watershed(~final_sim_map)
        scores = np.ones(len(masklets))
        print(masklets.shape)

        return masklets, scores
