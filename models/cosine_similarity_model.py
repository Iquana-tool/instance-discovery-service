import numpy as np
from PIL.Image import fromarray
from torch import Tensor

from app.schemas.inference import Request
from models.base_models import BaseModel
from models.encoders.dino import DinoModel, DinoModelType
from models.encoders.encoder import Encoder
from models.predictors.cosine_similarity_predictor import CosineSimilarityPredictor


class CosineSimilarityModel(BaseModel):
    def __init__(self, predictor: CosineSimilarityPredictor, backbone: Encoder):
        self.predictor = predictor
        self.backbone = backbone

    def process_request(self, image, request: Request):
        if isinstance(image, np.ndarray):
            image = fromarray(image)
        embedded_img: Tensor = self.backbone.embed_image(image=image)
        self.predictor.reset()
        # Seeds is a list of lists of indices that specify which pixels should be used as a seed. Each list represents
        # one object.
        for seed in request.seeds:
            self.predictor.add_seed_instance(embedded_img.flatten(end_dim=1)[seed])
        mask = self.predictor.predict(embedded_img)
        return mask.cpu().numpy(),  self.predictor.threshold


class Dino1000CosineHeMaxAgg(CosineSimilarityModel):
    def __init__(self):
        super().__init__(
            predictor=CosineSimilarityPredictor(
                memory_aggregation="none",
                similarity_aggregation="mean",
                similarity_redistribution_method="norm"),
            backbone=DinoModel(
                model_type=DinoModelType.VITS16,
                patch_size=16,
                image_size=1024,
            )
        )
