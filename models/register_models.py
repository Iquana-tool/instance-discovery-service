from models.cosine_similarity_model import Dino1000CosineHeMaxAgg
from models.geco_model import GeCoCompletion
from models.model_registry import ModelRegistry, ModelInfo, ModelLoader
from models.sam3 import SAM3Completion
from paths import *


def register_models(model_registry: ModelRegistry):
    """ This function registers all models in the MODEL_REGISTRY. You can extend it to add custom models. """
    model_registry.register_model(
        ModelInfo(
            identifier_str="cosine_sim",
            name="Cosine similarity predictor",
            description="A dual encoder decoder architecture using DINO v3 backbone with an image size of 1024 px. The "
                        "decoder uses cosine similarity with maximum similarity aggregation and histogram equalization "
                        "to find similar objects in the image. SAM is used to refine the proposed masks.",
            tags=["Experimental"],
            supports_refinement=False,
        ),
        ModelLoader(
            loader_function=Dino1000CosineHeMaxAgg
        )
    )
    model_registry.register_model(
        ModelInfo(
            identifier_str="sam3",
            name="SAM 3",
            description="SAM 3 is a unified foundation model for promptable segmentation in images and videos. "
                        "It can detect, segment, and track objects using text or visual prompts such as points, boxes, "
                        "and masks. Compared to its predecessor SAM 2, SAM 3 introduces the ability to exhaustively "
                        "segment all instances of an open-vocabulary concept specified by a short text phrase or "
                        "exemplars. Unlike prior work, SAM 3 can handle a vastly larger set of open-vocabulary prompts. "
                        "It achieves 75-80% of human performance on our new SA-CO benchmark which contains 270K unique "
                        "concepts, over 50 times more than existing benchmarks.",
            tags=["Meta AI"],
            supports_refinement=False,
        ),
        ModelLoader(
            loader_function=SAM3Completion,
            threshold=0.5,

        )
    )
    model_registry.register_model(
        ModelInfo(
            identifier_str='geco',
            name="GeCo",
            description="GeCo, a low-shot counter that achieves accurate object detection, segmentation, and "
                        "count estimation in a unified architecture. GeCo robustly generalizes the prototypes across "
                        "objects appearances through a novel dense object query formulation.",
            tags=["SOTA"],
            supports_refinement=False,
        ),
        ModelLoader(
            loader_function=GeCoCompletion,
        )
    )
