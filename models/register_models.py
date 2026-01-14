from models.dbscan_predictor import DBSCANPredictor, DBScanModel
from models.simple_thresholding_model import Dino1000CosineHeMaxAgg
from models.geco_model import GeCoCompletion
from models.knn_graph_predictor import KNNGraphPredictor
from models.model_registry import ModelRegistry, ModelInfo, ModelLoader
from models.sam3 import SAM3Completion


def register_models(model_registry: ModelRegistry):
    """ This function registers all models in the MODEL_REGISTRY. You can extend it to add custom models. """
    model_registry.register_model(
        ModelInfo(
            identifier_str="thresholding",
            name="Simple Thresholding Model",
            description="This model embeds the image using DINOv3 and then computes a heatmap of the cosine similarity "
                        "between provided exemplar embeddings and remaining embeddings. Objects are detected by "
                        "thresholding this heatmap and extracting connected components. ",
            tags=["Simple", "Experimental", "Not accurate"],
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
            tags=["Prompted Concept Segmentation","General Domain", "Meta AI"],
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
            tags=["Object Detection", "AI"],
            supports_refinement=False,
        ),
        ModelLoader(
            loader_function=GeCoCompletion,
        )
    )
    model_registry.register_model(
        ModelInfo(
            identifier_str='dbscan',
            name="DBSCAN Cluster Model",
            description="""
            A DBSCAN-based clustering model for self-supervised object discovery in high-dimensional embeddings (e.g., DINO, SAM).
            Uses density-based clustering to group similar regions, ideal for segmenting objects (e.g., penguins) without ground truth labels.
            Works well with exemplar-guided segmentation: provide 1+ seed masks to identify all instances of the target class.
            Optimized for static embeddings and supports fast inference for interactive applications.
            """,
            tags=[
                "experimental", "clustering", "self-supervised", "density-based",
                "interactive-ml", "segmentation", "embedding-analysis"
            ],
            supports_refinement=False,
        ),
        ModelLoader(
            loader_function=DBScanModel,  # Corrected to match your implementation
        )
    )

