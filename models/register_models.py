from models.few_shot_finetuning import AttentionFewShotModel
from models.geco_model import GeCoCompletion
from models.sam3 import SAM3Completion
from models.sansa_detection import SANSA
from models.watershed_dino import WatershedDINO
import logging

from iquana_toolbox.mlflow import MLFlowModelRegistry

logger = logging.getLogger(__name__)

MODEL_REGISTRY_CONFIG = [
    {
        "model_identifier": "sam3",
        "model_factory": lambda: SAM3Completion(threshold=0.5),
        "desc": "SAM 3 is a unified foundation model for promptable segmentation in images and videos. It supports text and visual prompts including points, boxes, and masks.",
        "tags": {
            "task": "instance-discovery",
            "status": "ready",
            "pretrained": "true",
            "trainable": "false",
            "finetunable": "true",
            "domain": "general",
            "publisher": "meta-ai",
        }
    },
    {
        "model_identifier": "sansa",
        "model_factory": lambda : SANSA(),
        "desc": "SANSA uses Segment Anything 2 features to provide few-shot object and part segmentation without fine-tuning SAM2 weights.",
        "tags": {
            "task": "few-shot-segmentation",
            "status": "ready",
            "pretrained": "true",
            "trainable": "false",
            "finetunable": "false",
            "domain": "general",
        }
    },
    {
        "model_identifier": "geco",
        "model_factory": lambda: GeCoCompletion(),
        "desc": "GeCo is a low-shot counting model that performs object detection, segmentation, and count estimation in one architecture.",
        "tags": {
            "task": "instance-discovery",
            "status": "ready",
            "pretrained": "true",
            "trainable": "false",
            "finetunable": "true",
            "domain": "general",
        }
    },
    {
        "model_identifier": "few-shot-attention",
        "model_factory": lambda: AttentionFewShotModel(),
        "desc": "Few-shot attention model trained on pseudo-labels generated with cosine similarity.",
        "tags": {
            "task": "instance-discovery",
            "status": "experimental",
            "pretrained": "true",
            "trainable": "false",
            "finetunable": "true",
            "domain": "general",
        }
    },
    {
        "model_identifier": "watershed-dino",
        "model_factory": lambda : WatershedDINO(),
        "desc": "Computes cosine similarity between exemplars and the image, then applies watershed to extract basins.",
        "tags": {
            "task": "instance-discovery",
            "status": "experimental",
            "pretrained": "true",
            "trainable": "false",
            "finetunable": "true",
            "domain": "general",
        }
    },
]


def register_models(model_registry: MLFlowModelRegistry):
    """Lazily register models only if they don't already exist in MLflow.

    This avoids expensive model instantiation at startup if the models are already
    registered in the MLflow registry.
    """
    for config in MODEL_REGISTRY_CONFIG:
        model_id = config["model_identifier"]

        if model_registry.check_registered(model_id):
            continue

        logger.info(f"Registering model '{model_id}' (not found in registry)...")
        try:
            model = config["model_factory"]()
            model_registry.register_model(
                model_identifier=model_id,
                model=model,
                desc=config["desc"],
                tags=config["tags"]
            )
            logger.info(f"Successfully registered model '{model_id}'.")
        except Exception as e:
            logger.error(f"Failed to register model '{model_id}': {e}")
            continue

