from fastapi import APIRouter
import torch


router = APIRouter()


@router.get("/health")
async def health_check():
    # Check Device
    if torch.cuda.is_available():
        device_status = f"cuda ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        device_status = "mps (Apple Silicon)"
    else:
        device_status = "cpu"

    return {
        "status": "ok",
        "device": device_status,
        "torch_version": torch.__version__
    }
