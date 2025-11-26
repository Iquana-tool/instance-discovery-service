from fastapi import status, HTTPException, Response

from app.routes import router
from app.schemas.inference import Request
from app.state import MODEL_CACHE, IMAGE_CACHE, MODEL_REGISTRY
from models.base_models import BaseModel


@router.post("/infer_instances")
async def infer_instances(request: Request):
    """ Infer instances from seed instances. """
    if not request.user_id in IMAGE_CACHE:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No uploaded image for user id not found. "
                                                                          "Make sure to upload an image to this service "
                                                                          "first before running inference.")
    image = IMAGE_CACHE.get(request.user_id)
    if not request.user_id in MODEL_CACHE:
        MODEL_CACHE.put(request.user_id, MODEL_REGISTRY.load_model(request.user_id))
    model: BaseModel = MODEL_CACHE.get(request.user_id)
    mask, score = model.process_request(image, request)

    # Convert the mask to raw bytes
    mask_bytes = mask.tobytes()

    # Return the raw bytes with metadata in headers
    return Response(
        content=mask_bytes,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": "attachment; filename=mask.bin",
            "X-Mask-Shape": f"{mask.shape[0]},{mask.shape[1]}",  # e.g., "256,256"
            "X-Mask-Dtype": str(mask.dtype),  # e.g., "uint8"
            "X-Score": str(score)  # Optional: Include the score
        }
    )

