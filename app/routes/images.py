from logging import getLogger

from fastapi import UploadFile, File

from app.routes import session_router
from app.state import IMAGE_CACHE
from util.image_loading import load_image_from_upload

logger = getLogger(__name__)


@session_router.post("/open_image/user_id={user_id}")
async def open_image(user_id: str, image: UploadFile = File(...)):
    """Endpoint to upload an image and an optional previous mask.
    This is a placeholder endpoint to demonstrate file upload functionality.
    In a real application, you might want to store the image and return an ID or URL.
    """
    image = load_image_from_upload(image)
    IMAGE_CACHE.set(user_id, image)
    return {
        "success": True,
        "message": f"Image uploaded successfully for user {user_id}.",
        "image_shape": image.shape
    }


@session_router.get("/focus_crop/min_x={min_x}&min_y={min_y}&max_x={max_x}&max_y={max_y}&user_id={user_id}")
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
    IMAGE_CACHE.set_focused_crop(user_id, min_x, min_y, max_x, max_y)
    return {
        "success": True,
        "message": f"Image cropped successfully for user {user_id}.",
    }


@session_router.get("/unfocus_crop/user_id={user_id}")
async def unfocus_crop(user_id: str):
    """Revert the cached image to the original uploaded image.
    :param user_id: Unique identifier for the user to retrieve their cached image.
    :return: Success message with new image shape or error message.
    """
    if user_id not in IMAGE_CACHE:
        return {"success": False, "message": "No image uploaded for this user. Please upload an image first."}
    IMAGE_CACHE.set_uncropped(user_id)
    return {
        "success": True,
        "message": f"Image reverted to original successfully for user {user_id}.",
    }


@session_router.get("/close_image/user_id={user_id}")
async def close_image(user_id: str):
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
