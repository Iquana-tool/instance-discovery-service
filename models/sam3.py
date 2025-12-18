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
        bboxes = request.get_bboxes(
            format="cxcywh",
            relative_coordinates=False,
            device="cpu",
            return_tensors=False,
            resize_to=None
        )
        bbox_labels = torch.ones(len(bboxes), dtype=torch.float32).unsqueeze(0)
        print(image.shape)
        # Preprocess the image and prompts
        inputs = self.processor(
            images=[image],
            input_boxes=[bboxes],
            input_boxes_labels=bbox_labels,
            return_tensors="pt"
        )
        inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        print(f"Found objects:\t{len(outputs['pred_masks'])}")
        # Post-process results
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.threshold,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]

        print(f"After postprocessing:\t{len(results['masks'])}")
        masks = results["masks"].cpu().numpy()
        print(masks.shape)
        scores = results["scores"].cpu().numpy()
        keep_ids = filter_seed_masks(request.get_combined_seed_mask(inputs.get("original_sizes").tolist()[0]), masks)
        print(f"After filtering: {len(keep_ids)}")
        return InstanceMasksResponse.from_masks(masks[keep_ids], scores[keep_ids])


