from logging import getLogger

from fastapi import APIRouter
from iquana_toolbox.schemas.service_requests import BaseImageRequest

from app.state import IMAGE_CACHE

router = APIRouter()
session_router = APIRouter(prefix="/annotation_session", tags=["annotation_session"])

logger = getLogger(__name__)


@session_router.post("/images/preload")
async def open_image(request: BaseImageRequest):
    """Endpoint to upload an image and an optional previous mask.
    This is a placeholder endpoint to demonstrate file upload functionality.
    In a real application, you might want to store the image and return an ID or URL.
    """
    IMAGE_CACHE.set(request.user_id, request.image)
    return {
        "success": True,
        "message": f"Image uploaded successfully for user {request.user_id}.",
    }


@session_router.get("/images/focus_crop")
async def focus_crop(
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        user_id: str,
):
    """Crop the uploaded image to the specified bounding box and update the cached image.
    :param min_x: Minimum x-coordinate of the bounding box.
    :param min_y: Minimum y-coordinate of the bounding box.
    :param max_x: Maximum x-coordinate of the bounding box.
    :param max_y: Maximum y-coordinate of the bounding box.
    :param user_id: Unique identifier for the user to retrieve their cached image.
    :return: Success message with new image shape or error message.
    """
    if user_id not in IMAGE_CACHE:
        return {"success": False, "message": "No image uploaded for this user. Please upload an image first."}
    #IMAGE_CACHE.set_focused_crop(user_id, min_x, min_y, max_x, max_y)
    return {
        "success": True,
        "message": f"Image cropped successfully for user {user_id}.",
    }


@session_router.get("/images/unfocus_crop")
async def unfocus_crop(user_id: str):
    """Revert the cached image to the original uploaded image.
    :param user_id: Unique identifier for the user to retrieve their cached image.
    :return: Success message with new image shape or error message.
    """
    if user_id not in IMAGE_CACHE:
        return {"success": False, "message": "No image uploaded for this user. Please upload an image first."}
    IMAGE_CACHE.unset_focused_crop(user_id)
    return {
        "success": True,
        "message": f"Image reverted to original successfully for user {user_id}.",
    }


@session_router.get("/images/clear_cache")
async def clear_cache_for_user(user_id: str):
    """Clear the cached image for the specified user.
    :param user_id: Unique identifier for the user to clear their cached image.
    :return: Success message or error message.
    """
    if user_id not in IMAGE_CACHE:
        return {"success": False, "message": "No image uploaded for this user. Please upload an image first."}
    IMAGE_CACHE.delete(user_id)
    return {
        "success": True,
        "message": f"Image cache cleared successfully for user {user_id}.",
    }
