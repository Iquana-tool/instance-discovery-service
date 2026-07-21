from typing import Any

import cv2
import numpy as np
import torch
from skimage.segmentation import watershed

from iquana_toolbox.ai.backbones.dinov3 import DINOv3Backbone, DEFAULT_DINOV3_MODEL
from iquana_toolbox.ai.base_classes import InstanceSuggestionModel, InstanceSuggestionModelInfo
from iquana_toolbox.schemas.networking.http.services import InstanceSuggestionRequest
from iquana_service_core import register_model

from models.similarity.cosine_similarity import CosineSimilarity
from paths import HF_ACCESS_TOKEN


@register_model
class WatershedDINO(InstanceSuggestionModel):
    model_info = InstanceSuggestionModelInfo(
        registry_key="watershed-dino",
        name="Watershed DINO",
        description=(
            "Computes cosine similarity between exemplars and the image using frozen DINOv3 "
            "patch features, then applies watershed to extract instance basins."
        ),
        usage_tip="Provide positive exemplar masks; works best on textured, densely packed instances.",
        tags={
            "task": "instance-suggestion",
            "status": "ready",
            "pretrained": "true",
            "finetunable": "false",
            "domain": "general",
            "publisher": "meta-ai",
        },
        status="ready",
        trainable=False,
    )

    # DINOv3Backbone wraps an HF model and can't be cloudpickled reliably.
    # Both attrs are stripped before pickling and rebuilt in ``load_context``.
    _unpicklable_attrs = ("backbone", "similarity")

    def __init__(
        self,
        image_size: int = 768,
        model_id: str = DEFAULT_DINOV3_MODEL,
        similarity=None,
        backbone=None,
    ):
        self.image_size = image_size
        self.model_id = model_id
        self._load_model(similarity=similarity, backbone=backbone)

    def _load_model(self, similarity=None, backbone=None):
        """Instantiate DINOv3 backbone + similarity head. Called at init and in load_context."""
        self.similarity = similarity if similarity is not None else CosineSimilarity(
            device="auto",
            memory_aggregation="none",
            similarity_aggregation="mean",
            similarity_redistribution_method="none",
        )
        self.backbone = backbone if backbone is not None else DINOv3Backbone(
            model_id=self.model_id,
            image_size=self.image_size,
            token=HF_ACCESS_TOKEN,
        )

    def load_context(self, context):
        self._load_model()

    def _embed_image(self, image: np.ndarray) -> torch.Tensor:
        """Run DINOv3 on ``image`` and return standardized features as ``(Hp, Wp, C)``."""
        pixel_values = self.backbone.preprocess(image)          # (1, 3, H, W)
        features = self.backbone(pixel_values)                  # (1, C, Hp, Wp)
        features = features.squeeze(0).permute(1, 2, 0)        # (Hp, Wp, C)
        mean = features.mean(dim=(0, 1), keepdim=True)
        std = features.std(dim=(0, 1), keepdim=True)
        return (features - mean) / (std + 1e-8)

    def predict(
        self,
        context: Any,
        model_input,
        params: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        request = model_input[0] if isinstance(model_input, list) else model_input

        # Extract DINOv3 patch features: (Hp, Wp, C).
        embedded_img = self._embed_image(request.image)
        Hp, Wp = embedded_img.shape[:2]

        # cv2.resize uses (width, height) convention.
        patch_wh = (Wp, Hp)

        # Average per-exemplar cosine-similarity maps at patch resolution.
        sim_maps = []
        for mask in request.positive_exemplar_masks:
            seed_mask_np = cv2.resize(mask.astype(np.uint8), patch_wh, interpolation=cv2.INTER_NEAREST).astype(bool)
            # Move the boolean mask to the same device as the feature tensor to avoid
            # CUDA/CPU device mismatches when the backbone runs on GPU.
            seed_mask_t = torch.from_numpy(seed_mask_np).to(embedded_img.device)
            self.similarity.reset()
            self.similarity.add_seed_instance(embedded_img[seed_mask_t])
            sim_map = self.similarity.get_similarity_map(embedded_img)
            sim_maps.append(sim_map)

        final_sim_map = torch.mean(torch.stack(sim_maps), dim=0).cpu().numpy()
        final_sim_map = (final_sim_map * 255).astype(np.uint8)

        label_map = watershed(~final_sim_map)  # (Hp, Wp) integer labels

        # Resize label map back to original image resolution.
        # cv2.resize only supports uint8/float32/float64 — cast via float32 to preserve
        # integer labels exactly (nearest interpolation copies values, no rounding needed).
        orig_h, orig_w = request.image.shape[:2]
        label_map_full = cv2.resize(
            label_map.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        ).astype(np.int32)

        # Split label map into individual (H, W) binary masks; skip background (0).
        instance_ids = np.unique(label_map_full)
        instance_ids = instance_ids[instance_ids != 0]
        if len(instance_ids) == 0:
            return np.empty((0, orig_h, orig_w), dtype=np.uint8), np.empty((0,), dtype=np.float32)

        masks = np.stack([(label_map_full == i).astype(np.uint8) for i in instance_ids])
        scores = np.ones(len(masks), dtype=np.float32)
        return masks, scores

    def train(self, request, **kwargs):
        raise NotImplementedError("WatershedDINO is not trainable.")
