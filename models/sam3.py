import numpy as np
import torch
from transformers.models.sam3 import Sam3Model, Sam3Processor

from app.schemas.inference import Request, InstanceMasksResponse
from models.base_models import BaseModel
from util.postprocess import filter_seed_masks


class SAM3Completion(BaseModel):
    def __init__(self,
                 threshold,
                 device="auto"):
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
        self.threshold = threshold

    def process_request(self, image, request: Request) -> InstanceMasksResponse:
        # Extract the prompts from the given instance masks
        bboxes = []
        for seed in request.seeds:
            mask = np.array(seed, dtype=bool)
            indices = np.argwhere(mask)
            x = np.min(indices[:, 1]).item()  / mask.shape[1]# Min x
            y = np.min(indices[:, 0]).item()  / mask.shape[0]# Min y
            w = np.max(indices[:, 1]).item() - x / mask.shape[1]  # width of the bbox
            h = np.max(indices[:, 0]).item() - y / mask.shape[0]  # height of the bbox
            bboxes.append([x, y, w, h])
        bbox_labels = torch.ones(len(bboxes), dtype=torch.float32).unsqueeze(0)
        # Preprocess the image and prompts
        inputs = self.processor(
            images=image,
            text="Corals",
            input_boxes=torch.tensor(bboxes).unsqueeze(0),
            input_boxes_labels=bbox_labels,
            return_tensors="pt"
        )
        inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process results
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.threshold,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]

        print(f"Found {len(results['masks'])} objects")
        masks = results["masks"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        keep_ids = filter_seed_masks(request.get_combined_seed_mask(inputs.get("original_sizes").tolist()[0]), masks)
        return InstanceMasksResponse(
            masks=masks[keep_ids].tolist(),
            scores=scores[keep_ids].tolist(),
        )


