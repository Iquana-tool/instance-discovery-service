from app.schemas.inference import Request
from models.base_models import BaseModel
from models.encoders.encoder import Encoder
from models.predictors.cosine_similarity_predictor import CosineSimilarityPredictor
from models.encoders.dino import DinoModel, DinoModelType
from models.predictors.predictor import SimilarityPredictor


class CosineSimilarityModel(BaseModel):
    def __init__(self, predictor: SimilarityPredictor, backbone: Encoder):
        self.predictor = predictor
        self.backbone = backbone

    def process_request(self, image, request: Request):
        embedded_img = self.backbone.embed_image(image=image)
        self.predictor.reset()
        # Seeds is a list of lists of indices that specify which pixels should be used as a seed. Each list represents
        # one object.
        for seed in request.seeds:
            self.predictor.add_seed_instance(embedded_img[seed])
        mask = self.predictor.predict(embedded_img)
        return mask


class Dino1000CosineHeMaxAgg(CosineSimilarityModel):
    def __init__(self):
        super().__init__(
            predictor=CosineSimilarityPredictor(
                memory_aggregation="none",
                similarity_aggregation="max",
                similarity_redistribution_method="he"),
            backbone=DinoModel(
                model_type=DinoModelType.VITS16PLUS,
                patch_size=12,
                image_size=1024,
            )
        )
