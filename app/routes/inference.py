from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.networking.http.services import CompletionRequest
from fastapi import APIRouter
from app.state import MODEL_REGISTRY
from models.base_models import BaseModel

router = APIRouter()
session_router = APIRouter(prefix="/annotation_session", tags=["annotation_session"])

@session_router.post("/run")
async def infer_instances(request: CompletionRequest):
    """ Infer instances from seed instances. """
    model: BaseModel = MODEL_REGISTRY.get_model_by_alias(request.model_registry_key, "latest")
    masklets, scores = model.process_request(request.image, request)
    print(masklets.shape)
    result = []
    for masklet, score in zip(masklets, scores):
        try:
            result.append(
                Contour.from_binary_mask(
                    masklet,
                    confidence=score
                )
            )
        except Exception as e:
            print(e)
    return {
        "success": True,
        "message": f"Detected {len(result)} objects for user {request.user_id}",
        "result": result,
    }

