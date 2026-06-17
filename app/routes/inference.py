from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.networking.http.services import InstanceDiscoveryRequest
from fastapi import APIRouter
from app.state import MODEL_REGISTRY

router = APIRouter()
session_router = APIRouter(prefix="/annotation_session", tags=["annotation_session"])


@session_router.post("/run")
async def infer_instances(request: InstanceDiscoveryRequest):
    """ Infer instances from seed instances. """
    model = MODEL_REGISTRY.get_model_by_alias(request.model_registry_key, "latest")
    # model is an MLflow PyFuncModel; predict(data) forwards to the model's
    # predict(context, model_input=data, params), returning (masklets, scores).
    masklets, scores = model.predict([request])
    result = []
    for masklet, score in zip(masklets, scores):
        try:
            result.append(Contour.from_binary_mask(masklet, confidence=score))
        except Exception as e:
            print(e)
    return {
        "success": True,
        "message": f"Detected {len(result)} objects for user {request.user_id}",
        "result": result,
    }
