import logging

from iquana_service_core import create_service_app

from app.state import MODEL_REGISTRY
from app.routes.inference import router as inference_router
from app.routes.inference import session_router as inference_session_router
from models.register_models import register_models

logger = logging.getLogger(__name__)


def _device_info() -> dict:
    """Service-specific health detail (kept here so service-core needs no torch)."""
    import torch

    if torch.cuda.is_available():
        device = f"cuda ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        device = "mps (Apple Silicon)"
    else:
        device = "cpu"
    return {"device": device, "torch_version": torch.__version__}


def create_app():
    return create_service_app(
        title="IQUANA Instance Discovery API",
        description="FastAPI backend for instance discovery / few-shot segmentation",
        task="instance-discovery",
        registry=MODEL_REGISTRY,
        register_models=register_models,
        inference_routers=[inference_router, inference_session_router],
        hf_login=True,
        health_extra=_device_info,
    )
