from logging import getLogger

from fastapi import HTTPException


from app.state import MODEL_REGISTRY, MODEL_CACHE
from fastapi import APIRouter


router = APIRouter()
session_router = APIRouter(prefix="/annotation_session", tags=["annotation_session"])
logger = getLogger(__name__)


@router.get("/models/all")
async def list_models():
    """ Lists all available models in the registry. """
    available_models = MODEL_REGISTRY.list_models(only_return_available=False)
    return {
        "success": True,
        "message": f"Retrieved {len(available_models)} available models.",
        "result": available_models}


@router.get("/models/all/available")
async def list_models():
    """ Lists all available models in the registry. """
    available_models = MODEL_REGISTRY.list_models(only_return_available=True)
    return {
        "success": True,
        "message": f"Retrieved {len(available_models)} available models.",
        "result": available_models}


@router.get("/models/{model_registry_key}")
async def list_models(model_registry_key: str):
    """ Lists all available models in the registry. """
    available_models = MODEL_REGISTRY.get_model_info(model_registry_key)
    return {
        "success": True,
        "message": f"Retrieved {len(available_models)} available models.",
        "result": available_models}


@session_router.get("/models/{model_registry_key}/preload")
async def load_model(model_registry_key: str, user_id: str):
    """ Loads a model into the cache if not already loaded. This is a convenience endpoint; models are loaded
        automatically when needed, but this can be called at the start
        of an annotation session to preload the model."""
    if MODEL_CACHE.check_if_loaded(user_id, model_registry_key):
        return {
            "success": True,
            "message": f"Model {model_registry_key} is already loaded in cache.",
            "model_id": model_registry_key
        }
    else:
        try:
            model = MODEL_REGISTRY.load_model(model_registry_key)
            MODEL_CACHE.put(user_id, model_registry_key, model)
            return {
                "success": True,
                "message": f"Model {model_registry_key} loaded successfully to cache.",
            }
        except Exception as e:
            raise e
            logger.error(e)
            if type(e) == KeyError:
                raise HTTPException(status_code=404, detail=f"Model {model_registry_key} is not registered. Please check available models.")
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
