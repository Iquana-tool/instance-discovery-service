from logging import getLogger
from fastapi import APIRouter, UploadFile, File

from app.state import IMAGE_CACHE
from util.image_loading import load_image_from_upload


router = APIRouter(prefix="/annotation_session", tags=["annotation_session"])
logger = getLogger(__name__)


@router.post("/open_image/user_uid={user_uid}")
async def open_image(user_uid: str, image: UploadFile = File(...)):
    """Endpoint to upload an image and an optional previous mask.
    This is a placeholder endpoint to demonstrate file upload functionality.
    In a real application, you might want to store the image and return an ID or URL.
    """
    image = load_image_from_upload(image)
    IMAGE_CACHE.set(user_uid, image)
    return {
        "success": True,
        "message": f"Image uploaded successfully for user {user_uid}.",
        "image_shape": image.shape
    }
