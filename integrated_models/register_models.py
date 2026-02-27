from iquana_toolbox.schemas.models import CompletionModel

from integrated_models.few_shot_finetuning import AttentionFewShotModel
from integrated_models.geco_model import GeCoCompletion
from integrated_models.model_registry import ModelRegistry, ModelLoader
from integrated_models.sam3 import SAM3Completion
from integrated_models.sansa_detection import SANSA
from integrated_models.simple_thresholding_model import Dino1000CosineHeMaxAgg


def register_models(model_registry: ModelRegistry):
    """ This function registers all integrated_models in the MODEL_REGISTRY. You can extend it to add custom integrated_models. """

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
            tags=["Prompted Concept Segmentation", "General Domain", "Meta AI"],
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
            registry_key='sansa',
            name="SANSA",
            description="SANSA unlocks the hidden semantics of Segment Anything 2, turning it into a powerful few-shot segmenter for both objects and parts."
                        "🚀 No fine-tuning of SAM2 weights."
                        "🧠 Fully promptable: points · boxes · scribbles · masks, making it ideal for real-world labeling."
                        "📈 State-of-the-art on few-shot object & part segmentation benchmarks."
                        "⚡ Lightweight: 3–5× faster, 4–5× smaller!",
            tags=["Few-Shot", "AI"],
            number_of_parameters=0,
            pretrained=True,
            finetunable=False,
            trainable=False,
        ),
        ModelLoader(
            loader_function=SANSA,
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
