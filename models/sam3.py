from typing import Any

import numpy as np
import torch
from transformers.models.sam3 import Sam3Model, Sam3Processor

from iquana_toolbox.ai.base_classes import InstanceDiscoveryModel, InstanceDiscoveryModelInfo
from iquana_toolbox.schemas.networking.http.services import InstanceDiscoveryRequest
from iquana_service_core import register_model

from paths import HF_ACCESS_TOKEN


@register_model
class SAM3Completion(InstanceDiscoveryModel):
    model_info = InstanceDiscoveryModelInfo(
        registry_key="sam3",
        name="SAM 3",
        description=(
            "SAM 3 is a unified foundation model for promptable segmentation in images "
            "and videos. It supports text and visual prompts including points, boxes, and masks."
        ),
        usage_tip="Provide one or more positive exemplar masks; an optional concept label guides text-prompted detection.",
        tags={
            "task": "instance-discovery",
            "status": "ready",
            "pretrained": "true",
            "finetunable": "true",
            "domain": "general",
            "publisher": "meta-ai",
        },
        status="ready",
        trainable=False,
    )

    def __init__(self, threshold: float = 0.5, device: str = "auto"):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.processor = Sam3Processor.from_pretrained("facebook/sam3", token=HF_ACCESS_TOKEN)
        self.model = Sam3Model.from_pretrained("facebook/sam3", token=HF_ACCESS_TOKEN).to(self.device)
        self.threshold = threshold

    def predict(
        self,
        context: Any,
        model_input: list[InstanceDiscoveryRequest],
        params: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        request = model_input[0] if isinstance(model_input, list) else model_input

        # Exemplar masks -> bounding boxes; an optional concept label adds a text prompt.
        bboxes = request.get_bboxes(format="xyxy", relative_coordinates=False)
        bbox_labels = torch.ones(len(bboxes), dtype=torch.float32).unsqueeze(0)

        inputs = self.processor(
            images=[request.image],
            text=request.concept.name if request.concept is not None else "visual",
            input_boxes=[bboxes],
            input_boxes_labels=bbox_labels,
            return_tensors="pt",
        )
        inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.threshold,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        masks = results["masks"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        return masks, scores

    def train(self, request, **kwargs):
        raise NotImplementedError("SAM3Completion is not trainable.")
