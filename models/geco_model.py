import numpy as np
import torch
from PIL.Image import fromarray
from geco.models.geco import GeCo
from schemas.contours import Contour
from schemas.service_requests import CompletionRequest
from torchvision import ops
from torchvision import transforms

from app.schemas.inference import Request, InstanceMasksResponse, BBoxesResponse
from models.base_models import BaseModel


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
        self.image_size = image_size
        self.num_objects = num_objects
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.kernel_dim = kernel_dim
        self.reduction = reduction
        self.model = GeCo.from_pretrained(
            image_size=self.image_size,
            num_objects=self.num_objects,
            emb_dim=self.emb_dim,
            num_heads=self.num_heads,
            kernel_dim=self.kernel_dim,
            train_backbone=False,
            reduction=self.reduction,
            zero_shot=False,
            inference_mode=True,
            return_masks=True)
        self.model.to(self.device)

    def process_request(self, image, request: CompletionRequest):
        if isinstance(image, np.ndarray):
            image = fromarray(image)
        image = image.resize((self.image_size, self.image_size))
        self.model.eval()
        with torch.no_grad():
            image_tensor = torch.from_numpy(np.array(image) / 255.).float().to(self.device)
            if image_tensor.shape[0] != (3, self.image_size, self.image_size):
                image_tensor = image_tensor.permute(2, 0, 1)
            image_tensor = image_tensor.unsqueeze(0)
            image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor)
            bboxes = request.get_bboxes(
                format="x1y1x2y2",
                relative_coordinates=True,
            )
            bboxes = torch.tensor(bboxes, dtype=torch.float32).to(self.device).unsqueeze(0)
            outputs, _, _, _ = self.model(image_tensor, bboxes)
            print("GeCo done")
            output = outputs[0]
            print("Original number of objects:\t", output["pred_boxes"].shape)
            selector = output['box_v'] > output['box_v'].max() / 12  # This is from the GeCo repo, not really sure what it does
            print("Selected:\t", torch.sum(selector).item())
            keep = ops.nms(output['pred_boxes'][selector],
                           output['box_v'][selector],
                           0.2)
            print("After NMS:\t", keep.shape[0])
            selected_masks = output['pred_masks'][selector.squeeze()]
            print(f"Selected masks:\t", selected_masks.shape)
            nms_masks = selected_masks[keep]
            print(f"NMS masks:\t", nms_masks.shape)
            masks = nms_masks.cpu().numpy()
            scores = ((output["scores"][selector])[keep]).cpu().tolist()
        return masks, scores