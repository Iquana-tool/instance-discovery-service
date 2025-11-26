from logging import getLogger

from fastapi import HTTPException

from app.routes import router, session_router
from app.state import MODEL_REGISTRY, MODEL_CACHE

logger = getLogger(__name__)


@router.get("/get_available_models")
async def list_models():
    """ Lists all available models in the registry. """
    available_models = MODEL_REGISTRY.list_models(only_return_available=True)
    return {
        "success": True,
        "message": f"Retrieved {len(available_models)} available models.",
        "available_models": available_models}


@session_router.get("/load_model/model_key={model_id}&user_id={user_id}")
async def load_model(model_id: int, user_id: int):
    """ Loads a model into the cache if not already loaded. This is a convenience endpoint; models are loaded
        automatically when needed, but this can be called at the start
        of an annotation session to preload the model."""
    if MODEL_CACHE.check_if_loaded(user_id):
        return {
            "success": True,
            "message": f"Model {model_id} is already loaded in cache.",
            "model_id": model_id
        }
    else:
        try:
            model = MODEL_REGISTRY.load_model(model_id)
            MODEL_CACHE.put(user_id, model)
            return {
                "success": True,
                "message": f"Model {model_id} loaded successfully to cache.",
            }
        except Exception as e:
            logger.error(e)
            if type(e) == KeyError:
                raise HTTPException(status_code=404, detail=f"Model {model_id} is not registered. Please check available models.")
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
