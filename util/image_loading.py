import cv2
import numpy as np
from fastapi import UploadFile


def load_image_from_upload(upload: UploadFile):
    """Load an image from an UploadFile object and return it as a PIL Image."""
    image_data = upload.file.read()
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    return image
