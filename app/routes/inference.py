from iquana_toolbox.schemas.contours import Contour
from schemas.service_requests import CompletionRequest

from app.state import MODEL_CACHE, IMAGE_CACHE, MODEL_REGISTRY
from models.base_models import BaseModel

router = APIRouter()
session_router = APIRouter(prefix="/annotation_session", tags=["annotation_session"])

@session_router.post("/completion")
async def infer_instances(request: CompletionRequest):
    """ Infer instances from seed instances. """
    if not request.user_id in IMAGE_CACHE:
        IMAGE_CACHE.set(request.user_id, request.image)
    image = IMAGE_CACHE.get(request.user_id)
    if not MODEL_CACHE.check_if_loaded(request.user_id, request.model_registry_key):
        MODEL_CACHE.put(request.user_id, request.model_registry_key, MODEL_REGISTRY.load_model(request.model_registry_key))
    model: BaseModel = MODEL_CACHE.get(request.user_id)
    masklets, scores = model.process_request(image, request)
    print(masklets.shape)
    result = [
        Contour.from_binary_mask(
            masklet,
            confidence=score
        ) for masklet, score in zip(masklets, scores)
    ]
    return {
        "success": True,
        "message": f"Detected {len(result)} objects for user {request.user_id}",
        "result": result,
    }

