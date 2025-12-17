import numpy as np
from fastapi import status, HTTPException, Response

from app.schemas.inference import Request
from app.state import MODEL_CACHE, IMAGE_CACHE, MODEL_REGISTRY
from models.base_models import BaseModel
from fastapi import APIRouter


router = APIRouter()
session_router = APIRouter(prefix="/annotation_session", tags=["annotation_session"])

@session_router.post("/infer_instances")
async def infer_instances(request: Request):
    """ Infer instances from seed instances. """
    if not request.user_id in IMAGE_CACHE:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No uploaded image for user id not found. "
                                                                          "Make sure to upload an image to this service "
                                                                          "first before running inference.")
    image = IMAGE_CACHE.get(request.user_id)
    if not MODEL_CACHE.check_if_loaded(request.user_id, request.model_key):
        MODEL_CACHE.put(request.user_id, request.model_key, MODEL_REGISTRY.load_model(request.model_key))
    model: BaseModel = MODEL_CACHE.get(request.user_id)
    response = model.process_request(image, request)
    return {
        "success": True,
        "message": f"Detected {response.n_objects} objects for user {request.user_id}",
        "response": response.model_dump(),
    }

