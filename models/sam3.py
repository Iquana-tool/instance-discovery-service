import torch
from iquana_toolbox.schemas.service_requests import CompletionRequest
from transformers.models.sam3 import Sam3Model, Sam3Processor

from models.base_models import BaseModel


class SAM3Completion(BaseModel):
    def __init__(self,
                 threshold,
                 device="auto"):
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
        self.threshold = threshold

    def process_request(self, image, request: CompletionRequest):
        # Extract the prompts from the given instance masks
        bboxes = request.get_bboxes(
            format="xyxy",
            relative_coordinates=False,
        )
        bbox_labels = torch.ones(len(bboxes), dtype=torch.float32).unsqueeze(0)
        # Preprocess the image and prompts
        inputs = self.processor(
            images=[image],
            text=request.concept.name if request.concept is not None else "visual",
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
        return masks, scores


