from typing import Any

import cv2
import numpy as np
import torch
from PIL.Image import fromarray
from skimage.segmentation import watershed

from iquana_toolbox.ai.base_classes import InstanceSuggestionModel, InstanceSuggestionModelInfo
from iquana_toolbox.schemas.networking.http.services import InstanceSuggestionRequest
from iquana_service_core import register_model

from models.encoders.dino_encoder import DinoModel, DinoModelType
from models.similarity.cosine_similarity import CosineSimilarity


@register_model
class WatershedDINO(InstanceSuggestionModel):
    model_info = InstanceSuggestionModelInfo(
        registry_key="watershed-dino",
        name="Watershed DINO",
        description=(
            "Computes cosine similarity between exemplars and the image, then applies "
            "watershed to extract basins."
        ),
        usage_tip="Provide positive exemplar masks; experimental, works best on textured, densely packed instances.",
        tags={
            "task": "instance-suggestion",
            "status": "experimental",
            "pretrained": "true",
            "finetunable": "true",
            "domain": "general",
        },
        status="not_ready",
        trainable=False,
    )

    # Live torch backbones can't be cloudpickled; they are stripped from the pickle
    # and rebuilt from source in ``load_context``.
    _unpicklable_attrs = ("backbone", "similarity")

    def __init__(self, max_image_size=1024, similarity=None, backbone=None):
        self.max_image_size = [max_image_size, max_image_size] if isinstance(max_image_size, int) else max_image_size
        self._load_model(similarity=similarity, backbone=backbone)

    def _load_model(self, similarity=None, backbone=None):
        """Instantiate the DINO backbone + similarity head. Reused on first init and
        when MLflow rebuilds the model in ``load_context`` after unpickling."""
        self.similarity = similarity if similarity is not None else CosineSimilarity(
            device="auto",
            memory_aggregation="none",
            similarity_aggregation="mean",
            similarity_redistribution_method="none",
        )
        self.backbone = backbone if backbone is not None else DinoModel(
            device="auto",
            model_type=DinoModelType.VITL16,
            patch_size=16,
            image_size=1024,
        )

    def load_context(self, context):
        self._load_model()

    def predict(
        self,
        context: Any,
        model_input,
        params: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        request = model_input[0] if isinstance(model_input, list) else model_input
        image = request.image

        # Preprocess image
        if isinstance(image, np.ndarray):
            image = fromarray(image)
        # Resize to a fixed size to control computational effort as images can get large.
        image = image.resize(self.max_image_size)

        # Embed the image and standardize embeddings.
        embedded_img = self.backbone.embed_image(image=image, standardize=True)

        # Combine all seed masks into a single binary mask.
        combined_seed_mask = request.combined_exemplar_mask.astype(np.uint8)
        combined_seed_mask = cv2.resize(combined_seed_mask, self.max_image_size)

        # Process seeds separately and average similarity maps.
        sim_maps = []
        for mask in request.positive_exemplar_masks:
            seed_mask = np.array(fromarray(mask).resize(self.max_image_size)).astype(bool)
            self.similarity.reset()
            self.similarity.add_seed_instance(embedded_img[seed_mask])
            sim_map = self.similarity.get_similarity_map(embedded_img)
            sim_maps.append(sim_map)
        final_sim_map = torch.mean(torch.stack(sim_maps), dim=0).cpu().numpy()
        final_sim_map = (final_sim_map * 255).astype(np.uint8)

        # Watershed.
        masklets = watershed(~final_sim_map)
        scores = np.ones(len(masklets))
        return masklets, scores

    def train(self, request, **kwargs):
        raise NotImplementedError("WatershedDINO is not trainable.")
