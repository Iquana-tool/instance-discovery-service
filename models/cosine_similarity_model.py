from typing import Union

import numpy as np
from PIL.Image import fromarray, Image
from torch import Tensor
import plotly.express as px
import plotly.graph_objects as go
from app.schemas.inference import Request
from models.base_models import BaseModel
from models.encoders.dino import DinoModel, DinoModelType
from models.encoders.encoder import Encoder
from models.predictors.cosine_similarity_predictor import CosineSimilarityPredictor


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

    def process_request(self, image, request: Request):
        if isinstance(image, np.ndarray):
            image = fromarray(image)
        image = image.resize(self.max_image_size)
        print("Embedding image!")
        embedded_img: Tensor = self.backbone.embed_image(image=image)
        print(f"Embedded image shape: {embedded_img.shape}")
        self.predictor.reset()
        for seed in request.seeds:
            seed_mask = np.array(seed, dtype=np.bool)
            seed_mask = np.array(fromarray(seed_mask).resize(self.max_image_size))
            self.predictor.add_seed_instance(embedded_img[seed_mask])
        print("Computing cosine similarity...")
        mask = self.predictor.predict(embedded_img)
        print(f"Returning thresholded map. Mask shape: {mask.shape}")
        fig = px.imshow(mask * 255)
        fig.show()
        return mask.cpu().numpy(),  self.predictor.threshold


class Dino1000CosineHeMaxAgg(CosineSimilarityModel):
    def __init__(self):
        super().__init__(
            max_image_size=256,
            predictor=CosineSimilarityPredictor(
                device="auto",
                memory_aggregation="none",
                similarity_aggregation="mean",
                similarity_redistribution_method="norm"),
            backbone=DinoModel(
                device="auto",
                model_type=DinoModelType.VITS16PLUS,
                patch_size=16,
                image_size=256,
            )
        )
