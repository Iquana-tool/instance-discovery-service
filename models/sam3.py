from typing import Any

import numpy as np
import torch
from transformers.models.sam3 import Sam3Model, Sam3Processor

from iquana_toolbox.ai.base_classes import InstanceSuggestionModel, InstanceSuggestionModelInfo
from iquana_toolbox.schemas.networking.http.services import InstanceSuggestionRequest
from iquana_service_core import register_model

from paths import HF_ACCESS_TOKEN


@register_model
class SAM3Completion(InstanceSuggestionModel):
    model_info = InstanceSuggestionModelInfo(
        registry_key="sam3",
        name="SAM 3",
        description=(
            "SAM 3 is a unified foundation model for promptable segmentation in images "
            "and videos. It supports text and visual prompts including points, boxes, and masks."
        ),
        usage_tip=(
            "Provide one or more positive exemplar masks; an optional concept label guides "
            "text-prompted detection. Tune `threshold` (detection sensitivity, default 0.3 -- "
            "lower finds more) and `mask_threshold` (mask binarization) per request via params."
        ),
        tags={
            "task": "instance-suggestion",
            "status": "ready",
            "pretrained": "true",
            "finetunable": "true",
            "domain": "general",
            "publisher": "meta-ai",
            "threshold": "0.3",
        },
        status="ready",
        trainable=False,
    )

    # Live HF objects can't be cloudpickled (transformers attaches ContextVar-backed
    # forward hooks). They are stripped from the pickle and rebuilt in ``load_context``.
    _unpicklable_attrs = ("model", "processor")

    # SAM 3 scores are sigmoid(class) * sigmoid(presence) -- a product of two
    # probabilities, so they sit low. The HF-calibrated default is 0.3; anything
    # near 0.5 over-filters and the model "finds almost nothing".
    def __init__(self, threshold: float = 0.3, mask_threshold: float = 0.5, device: str = "auto"):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.threshold = threshold
        # Binarization point for each kept instance's mask. Note: 0.0 marks the whole
        # frame as foreground -- raise toward 0.5 for tight per-instance masks.
        self.mask_threshold = mask_threshold
        self._load_model()

    def _load_model(self):
        """Load the (pretrained) SAM 3 weights from the Hub. Reused on first init and
        when MLflow rebuilds the model in ``load_context`` after unpickling."""
        self.processor = Sam3Processor.from_pretrained("facebook/sam3", token=HF_ACCESS_TOKEN)
        self.model = Sam3Model.from_pretrained("facebook/sam3", token=HF_ACCESS_TOKEN).to(self.device)

    def load_context(self, context):
        self._load_model()

    def predict(
        self,
        context: Any,
        model_input,
        params: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        request = model_input[0] if isinstance(model_input, list) else model_input

        # Per-request overrides fall back to the values set at construction.
        params = params or {}
        threshold = params.get("threshold", self.threshold)
        mask_threshold = params.get("mask_threshold", self.mask_threshold)

        # Positive exemplar masks -> boxes with label 1; negative exemplars -> label 0.
        # SAM 3's geometry encoder uses these labels to push/pull the concept, so
        # dropping negatives discards a signal the caller explicitly provided.
        bboxes = request.get_bboxes(format="xyxy", relative_coordinates=False)
        labels = [1] * len(bboxes)
        for negative in request.negative_exemplars or []:
            bboxes.append(negative.get_as_bbox(relative_coords=False))
            labels.append(0)

        # Labels must be a LongTensor of shape (batch, num_boxes); an optional
        # concept label adds a text prompt, otherwise the "visual" token is used.
        bbox_labels = torch.tensor([labels], dtype=torch.int64)

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
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        masks = results["masks"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        return masks, scores

    def train(self, request, **kwargs):
        raise NotImplementedError("SAM3Completion is not trainable.")
