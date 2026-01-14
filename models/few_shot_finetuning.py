from typing import Union, Literal

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL.Image import fromarray

from app.schemas.inference import InstanceMasksResponse
from models.base_models import BaseModel
from models.encoders.dino_encoder import DinoModel, DinoModelType
from models.encoders.encoder_base_class import Encoder
from util.misc import get_device_from_str
from util.postprocess import extract_masklets


class FewShotPatchLevelModel(BaseModel):
    def __init__(
            self,
            backbone: Encoder | None = None,
            max_image_size: Union[int, list[int]] = 512,
            head_hidden_dim: int = 64,
            num_epochs: int = 20,
            lr: float = 0.001,
            device: Literal["auto", "cpu", "cuda"] = "auto",
    ):
        super().__init__()
        self.device = get_device_from_str(device)
        if backbone is None:
            self.backbone = DinoModel(
                device=self.device,
                model_type=DinoModelType.VITL16,
                patch_size=16,
                image_size=1024,
            )
        else:
            self.backbone = backbone.to(self.device)
        if type(max_image_size) == int:
            self.max_image_size = [max_image_size, max_image_size]
        else:
            self.max_image_size = max_image_size

        # Define a small MLP head for few-shot fine-tuning
        embedding_dim = self.backbone.embedding_dim  # Assume this is available
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.num_epochs = num_epochs
        self.lr = lr

    def _prepare_training_data(self, embedded_img, positive_masks, negative_masks):
        # Flatten masks and embeddings
        positive_pixels = []
        negative_pixels = []
        for mask in positive_masks:
            mask = np.array(mask, dtype=np.bool)
            mask = np.array(fromarray(mask).resize(self.max_image_size))
            positive_pixels.append(embedded_img[mask])
        for mask in negative_masks:
            mask = np.array(mask, dtype=np.bool)
            mask = np.array(fromarray(mask).resize(self.max_image_size))
            negative_pixels.append(embedded_img[mask])

        # Stack all positive and negative pixels
        positive_pixels = torch.cat(positive_pixels, dim=0)
        if negative_pixels:
            negative_pixels = torch.cat(negative_pixels, dim=0)
        else:
            negative_pixels = torch.empty((0, embedded_img.shape[-1]), device=embedded_img.device)

        # Create labels: 1 for positive, 0 for negative
        X = torch.cat([positive_pixels, negative_pixels], dim=0)
        y = torch.cat(
            [
                torch.ones(positive_pixels.shape[0], 1),
                torch.zeros(negative_pixels.shape[0], 1),
            ],
            dim=0,
        )
        return X.to(self.device), y.to(self.device)

    def _train_head(self, X, y):
        # Train the head using the prepared data
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.head.parameters(), lr=self.lr)

        # Send everything to the device
        self.head.to(self.device)
        criterion.to(self.device)

        # Train here
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            outputs = self.head(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

    def process_request(self, image, request):
        # Preprocess image
        if isinstance(image, np.ndarray):
            image = fromarray(image)
        image = image.resize(self.max_image_size)

        # Embed the image and standardize embeddings
        embedded_img = self.backbone.embed_image(image=image, standardize=True)

        # Prepare training data from positive and negative masks
        X, y = self._prepare_training_data(
            embedded_img, request.positive_masks, request.negative_masks
        )

        # Train the head
        self._train_head(X, y)

        # Inference: score all pixels using the trained head
        with torch.no_grad():
            h, w = embedded_img.shape[0], embedded_img.shape[1]
            embedded_img_flat = embedded_img.reshape(-1, embedded_img.shape[-1])
            scores = self.head(embedded_img_flat)
            scores = scores.reshape(h, w).cpu().numpy()

        # Convert scores to 0-255 range for thresholding
        scores = (scores * 255).astype(np.uint8)


        # Adaptive thresholding: use median of scores under positive masks
        combined_seed_mask = request.get_combined_seed_mask(self.max_image_size)
        threshold = np.median(scores[combined_seed_mask]).item() - 1
        _, thresholded = cv2.threshold(scores, threshold, 255, cv2.THRESH_BINARY)
        # Extract masklets
        masklets, scores = extract_masklets(thresholded, scores)

        return InstanceMasksResponse.from_masks(masklets, scores, request)
