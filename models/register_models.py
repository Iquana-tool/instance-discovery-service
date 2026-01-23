from schemas.models import CompletionModel

from models.few_shot_finetuning import AttentionFewShotModel
from models.geco_model import GeCoCompletion
from models.model_registry import ModelRegistry, ModelLoader
from models.sam3 import SAM3Completion
from models.simple_thresholding_model import Dino1000CosineHeMaxAgg


def register_models(model_registry: ModelRegistry):
    """ This function registers all models in the MODEL_REGISTRY. You can extend it to add custom models. """
    model_registry.register_model(
        CompletionModel(
            registry_key="thresholding",
            name="Simple Thresholding Model",
            description="This model embeds the image using DINOv3 and then computes a heatmap of the cosine similarity "
                        "between provided exemplar embeddings and remaining embeddings. Objects are detected by "
                        "thresholding this heatmap and extracting connected components. ",
            tags=["fast", "experimental", "pixel-level"],
            number_of_parameters=0,
            pretrained=True,
            finetunable=False,
            trainable=False,
        ),
        ModelLoader(
            loader_function=Dino1000CosineHeMaxAgg
        )
    )
    model_registry.register_model(
        CompletionModel(
            registry_key="sam3",
            name="SAM 3",
            description="SAM 3 is a unified foundation model for promptable segmentation in images and videos. "
                        "It can detect, segment, and track objects using text or visual prompts such as points, boxes, "
                        "and masks. Compared to its predecessor SAM 2, SAM 3 introduces the ability to exhaustively "
                        "segment all instances of an open-vocabulary concept specified by a short text phrase or "
                        "exemplars. Unlike prior work, SAM 3 can handle a vastly larger set of open-vocabulary prompts. "
                        "It achieves 75-80% of human performance on our new SA-CO benchmark which contains 270K unique "
                        "concepts, over 50 times more than existing benchmarks.",
            tags=["Prompted Concept Segmentation","General Domain", "Meta AI"],
            number_of_parameters=0,
            pretrained=True,
            finetunable=True,
            trainable=False,
        ),
        ModelLoader(
            loader_function=SAM3Completion,
            threshold=0.5,
        )
    )
    model_registry.register_model(
        CompletionModel(
            registry_key='geco',
            name="GeCo",
            description="GeCo, a low-shot counter that achieves accurate object detection, segmentation, and "
                        "count estimation in a unified architecture. GeCo robustly generalizes the prototypes across "
                        "objects appearances through a novel dense object query formulation.",
            tags=["Object Detection", "AI"],
            number_of_parameters=0,
            pretrained=True,
            finetunable=True,
            trainable=False,
        ),
        ModelLoader(
            loader_function=GeCoCompletion,
        )
    )
    model_registry.register_model(
        CompletionModel(
            registry_key='fs_attention_spatial',
            name="Few-Shot Spatial Attention Model",
            description="""
            A few-shot instance discovery model that utilizes a Cross-Attention mechanism 
            with Learned Spatial Positional Encodings on top of DINO features. Unlike 
            standard MLPs, this model considers the relative spatial layout of patches 
            and uses a Memory Bank of exemplars to score the entire image. Optimized 
            with Focal Loss to handle extreme class imbalance between background and objects.
            """,
            tags=[
                "few-shot", "spatial-aware", "attention", "dino-v2", "instance-discovery"
            ],
            # The head is roughly 128 (hidden) * 1024 (DINO dim) + Attention weights
            # Usually falls under < 1.5M parameters
            number_of_parameters=1500000,
            pretrained=True,
            finetunable=True,
            trainable=True,
        ),
        ModelLoader(
            loader_function=AttentionFewShotModel,
        )
    )
