from typing import Literal

import cv2
import numpy as np
import torch
from PIL.Image import Image
from iquana_toolbox.schemas.service_requests import CompletionRequest
from torchvision import transforms

from integrated_models.base_models import BaseModel


class SANSA(BaseModel):
    def __init__(self):
        super().__init__()
        self._transform = transforms.Compose([transforms.Resize(size=(640, 640)),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                              ])
        self.model = torch.hub.load(
            r'C:\Users\role01-admin\PycharmProjects\SANSA',
            'sansa',
            trust_repo=True,
            source="local",
            device=self.device
        )

    def format_prompt(self, n_shots: int, prompt_input, prompt_type: Literal['mask', 'point', 'box'] = 'mask'):
        """
        Format prompt to be fed to the SANSA model. Alternatively, import as 'from util.demo_sansa import format_prompt'
        """
        assert prompt_type in ['mask', 'point', 'box']
        prompt_dict = {0: {}, 'shots': n_shots}
        prompt_d = prompt_input
        if prompt_type == 'point':
            pts = torch.as_tensor(prompt_input, dtype=torch.float32, device=self.device).view(-1, 2)
            prompt_d = {'point_coords': pts.view(1, -1, 2),
                        'point_labels': torch.ones(1, pts.shape[0], dtype=torch.int32, device=self.device)}
        elif prompt_type == 'box':
            b = torch.as_tensor(prompt_input, dtype=torch.float32, device=self.device).view(-1, 4)
            x0y0 = torch.minimum(b[:, :2], b[:, 2:])
            x1y1 = torch.maximum(b[:, :2], b[:, 2:])
            point_coords = torch.stack([x0y0, x1y1], dim=1).view(1, -1, 2)
            n = point_coords.shape[1] // 2
            point_labels = torch.tensor([2, 3], dtype=torch.int32, device=self.device).repeat(1, n)
            prompt_d = {'point_coords': point_coords, 'point_labels': point_labels}
        prompt_dict[0][0] = {'prompt_type': prompt_type, 'prompt': prompt_d}
        return prompt_dict

    def process_request(self, image, request: CompletionRequest) -> tuple[np.ndarray, np.ndarray]:
        # Support and Query Image are the same for Intra Image Instance Discovery
        support_img = self._transform(image)

        # SANSA requires a "video" (support and query images stacked)
        video = torch.stack([support_img, support_img], dim=0)[None].to(self.device)

        # SANSA can only be prompted with one mask, so we pass the combined mask
        prompt = self.format_prompt(n_shots=1, prompt_input=request.combined_exemplar_mask)

        with torch.no_grad():
            out = self.model(video, prompt)

        # SANSA outputs only one mask, we need to find the objects on this mask
        density_mask = out["pred_masks"][1].sigmoid()

        masks = []  # Binary object masks
        scores = []  # Scores of the mask, can be confidence values or mock up

        # Convert sigmoid output to binary mask (0 or 255)
        # Using 0.5 as a standard threshold for object presence
        binary_mask = (density_mask.cpu().numpy() > 0.5).astype(np.uint8) * 255

        # Find external contours (individual objects)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            # Create a blank mask for each individual object
            m = np.zeros_like(binary_mask)
            cv2.drawContours(m, [cnt], -1, 255, thickness=cv2.FILLED)

            # Normalize to binary 0/1 for the output
            masks.append((m > 0).astype(np.uint8))

            # For scores, we can take the average sigmoid value within the contour
            # as a confidence metric
            cnt_mask = (m > 0)
            score = density_mask[cnt_mask].mean().item()
            scores.append(score)

        return np.array(masks), np.array(scores)




