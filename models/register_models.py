from models.cosine_similarity_model import Dino1000CosineHeMaxAgg
from models.model_registry import ModelRegistry, ModelInfo, ModelLoader
from paths import *


def register_models(model_registry: ModelRegistry):
    """ This function registers all models in the MODEL_REGISTRY. You can extend it to add custom models. """
    model_registry.register_model(
        ModelInfo(
            identifier_str="dino_1000_cosine_he_max_agg",
            name="Cosine similarity predictor with dino backbone",
            description="A dual encoder decoder architecture using DINO v3 backbone with an image size of 1000 px. The "
                        "decoder uses cosine similarity with maximum similarity aggregation and histogram equalization "
                        "to find similar objects.",
            tags=["Experimental"],
            supports_refinement=False,
        ),
        ModelLoader(
            loader_function=Dino1000CosineHeMaxAgg
        )
    )