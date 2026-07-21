"""INSID3-Instance: training-free instance suggestion from DINOv3.

A replacement for the conceptually-flawed ``WatershedDINO`` (which watershedded a *class*
similarity map -- a semantic signal with no instance structure). This model uses exemplar
similarity only as a **gate**, then discovers instances with the **recursive coarse-to-fine
cascade** (:func:`models.insid3.cascade.discover_instances`): re-embed → connected-components
fixed point → CLS re-identification, zooming in until each instance is isolated. This handles
small / densely-packed instances that a single flat pass cannot (the exemplar gets its own
object-centered crop, so it is not lost on the coarse full-image grid).

The heavy lifting lives in :mod:`models.insid3` (one module per stage) so the companion
notebook ``notebooks/insid3_recursive_discovery.ipynb`` can visualize every step. This wrapper
is the thin MLflow ``pyfunc`` adapter that the service registry loads.
"""

from typing import Any

import numpy as np

from iquana_toolbox.ai.backbones.dinov3 import DEFAULT_DINOV3_MODEL, DINOv3Backbone
from iquana_toolbox.ai.base_classes import InstanceSuggestionModel, InstanceSuggestionModelInfo
from iquana_toolbox.schemas.networking.http.services import InstanceSuggestionRequest
from iquana_service_core import register_model

from models.insid3 import cascade, InSID3Params
from paths import HF_ACCESS_TOKEN

# Keys in the request ``params`` dict forwarded to ``cascade.discover_instances`` as kwargs.
# (Everything else in ``params`` is consumed by ``InSID3Params`` for the gate/split stages.)
_CASCADE_KWARGS = (
    "max_depth", "min_crop", "pad_frac", "shrink_stop", "cls_threshold", "clump_area_factor",
    "min_instance_area", "embed_batch_size", "max_total_embeds", "split_mode", "border_mode",
    "prototype_mode", "n_prototypes",
)


@register_model
class InSID3Instance(InstanceSuggestionModel):
    model_info = InstanceSuggestionModelInfo(
        registry_key="insid3-instance",
        name="INSID3-Instance",
        description=(
            "Training-free instance suggestion from frozen DINOv3 features. Uses exemplar "
            "similarity as a gate, then a recursive coarse-to-fine cascade: re-embed each crop, "
            "split by connected components, and zoom in until every instance is isolated and "
            "re-identifies the exemplar (CLS token). Handles small / densely-packed instances "
            "and separates same-class instances that a single pass merges."
        ),
        usage_tip=(
            "Provide one or more positive exemplar masks (negatives optional). Tune via params: "
            "gate_threshold (class sensitivity), cls_threshold (re-identification bar; lower if "
            "instances are dropped), border_mode in {proto, otsu, static} (mask border), "
            "shrink_stop / max_depth (zoom behaviour), min_instance_area (speck filter), "
            "max_total_embeds (compute cap). Small objects are supported -- the exemplar gets "
            "its own object-centered crop, so it is not lost on the coarse grid."
        ),
        tags={
            "task": "instance-suggestion",
            "status": "ready",
            "pretrained": "true",
            "finetunable": "false",
            "domain": "general",
            "publisher": "dfki",
        },
        status="ready",
        trainable=False,
    )

    # DINOv3Backbone wraps a live HF model that can't be cloudpickled reliably; it is
    # stripped before pickling and rebuilt in ``load_context`` (see BaseModel).
    _unpicklable_attrs = ("backbone",)

    def __init__(self, image_size: int = 1024, model_id: str = DEFAULT_DINOV3_MODEL, backbone=None):
        # Default to a larger input than WatershedDINO (768): instance separation is
        # patch-limited, and coral/dense scenes need a finer grid.
        self.image_size = image_size
        self.model_id = model_id
        self._load_model(backbone=backbone)

    def _load_model(self, backbone=None):
        self.backbone = backbone if backbone is not None else DINOv3Backbone(
            model_id=self.model_id,
            image_size=self.image_size,
            token=HF_ACCESS_TOKEN,
        )

    def load_context(self, context):
        self._load_model()

    def predict(
        self,
        context: Any,
        model_input,
        params: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        request: InstanceSuggestionRequest = model_input[0] if isinstance(model_input, list) else model_input
        params = params or {}

        # Defaults mirror the working notebook config. ``min_instance_area`` scales with the
        # exemplar (5 % of its area) like the notebook, so specks are dropped but real instances
        # kept -- the cascade default of 16 keeps everything, the notebook value filters.
        exemplar_area = max((int(m.sum()) for m in request.positive_exemplar_masks), default=0)
        cascade_defaults = dict(
            max_depth=8, min_crop=64, pad_frac=0.08, shrink_stop=0.9,
            cls_threshold=0.5, clump_area_factor=1.5,
            min_instance_area=max(16, int(0.05 * exemplar_area)),
            embed_batch_size=8, max_total_embeds=512,
            split_mode="watershed", border_mode="otsu",
            # "all" matches the notebook. "cluster" smooths the bank -> fewer patches gated at the
            # same gate_threshold, so it needs a LOWER gate_threshold (else it can return nothing).
            prototype_mode="all", n_prototypes=9,
        )
        # Request params override the defaults once the backend forwards them.
        cascade_defaults.update({k: params[k] for k in _CASCADE_KWARGS if k in params})

        stage_params = InSID3Params(gate_threshold=0.45, cluster_tau=0.18, marker_mode="hybrid")
        for key, value in params.items():
            if hasattr(stage_params, key):
                setattr(stage_params, key, value)

        instances, _ = cascade.discover_instances(
            backbone=self.backbone,
            image=request.image,
            exemplar_masks=request.positive_exemplar_masks,
            negative_masks=request.negative_exemplar_masks,
            params=stage_params,
            **cascade_defaults,
        )

        h, w = request.image.shape[:2]
        if not instances:
            return np.empty((0, h, w), dtype=np.uint8), np.empty((0,), dtype=np.float32)
        masks = np.stack([d.mask for d in instances]).astype(np.uint8)
        scores = np.asarray([d.cls_score for d in instances], dtype=np.float32)
        return masks, scores

    def train(self, request, **kwargs):
        raise NotImplementedError("INSID3-Instance is training-free.")
