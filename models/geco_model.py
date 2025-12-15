import torch

from app.schemas.inference import Request, InstanceMasksResponse, BBoxesResponse
from models.base_models import BaseModel
from geco.models.geco import GeCo


class GeCoCompletion(BaseModel):
    def __init__(self,
                 image_size: int = 1024,
                 num_objects: int = 10,
                 emb_dim: int = 256,
                 num_heads: int = 8,
                 kernel_dim: int = 1,
                 reduction: int = 16,
                 device: str = 'auto',):
        super().__init__()
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        self.model = GeCo.from_pretrained(
            image_size=image_size,
            num_objects=num_objects,
            emb_dim=emb_dim,
            num_heads=num_heads,
            kernel_dim=kernel_dim,
            train_backbone=False,
            reduction=reduction,
            zero_shot=False,
            inference_mode=True,
            return_masks=True)
        self.model.to(self.device)

    def process_request(self, image, request: Request) -> InstanceMasksResponse | BBoxesResponse:
        pass
