from typing import Union, Literal

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from PIL.Image import fromarray
from schemas.contours import Contour
from schemas.service_requests import CompletionRequest
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.nn.conv import GATConv
from tqdm import tqdm

from app.schemas.inference import InstanceMasksResponse
from models.base_models import BaseModel
from models.encoders.dino_encoder import DinoModel, DinoModelType
from models.encoders.encoder_base_class import Encoder
from util.debug import debug_show_image
from util.misc import get_device_from_str
from util.postprocess import extract_masklets


class FewShotPatchLevelModel(BaseModel):
    """ A small MLP head that trains to predict whether a pixel belongs to the same class or not. """

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

    def process_request(self, image, request: CompletionRequest):
        # Preprocess image
        if isinstance(image, np.ndarray):
            image = fromarray(image)
        image = image.resize(self.max_image_size)

        # Embed the image and standardize embeddings
        embedded_img = self.backbone.embed_image(image=image, standardize=True)

        # Prepare training data from positive and negative masks
        X, y = self._prepare_training_data(
            embedded_img,
            request.positive_exemplar_masks,
            request.negative_exemplar_masks,
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
        combined_seed_mask = cv2.resize(request.combined_exemplar_mask, self.max_image_size)
        threshold = np.median(scores[combined_seed_mask]).item() - 1
        _, thresholded = cv2.threshold(scores, threshold, 255, cv2.THRESH_BINARY)
        # Extract masklets
        masklets, scores = extract_masklets(thresholded, scores)

        return [
            Contour.from_binary_mask(masklet,
                                     label_id=None,
                                     added_by=request.model_registry_key)
            for masklet in masklets
        ]


class SpatialFewShotPatchLevelModel(BaseModel):
    """A model that predicts pixel class with spatial relationships using a graph attention layer."""
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

        # Define a graph attention layer for spatial relationships
        embedding_dim = self.backbone.embedding_dim
        self.gat_layer = GATConv(
            in_channels=embedding_dim,
            out_channels=head_hidden_dim,
            heads=4,  # Use 4 attention heads
            concat=True,
        )

        # Define a small MLP head for few-shot fine-tuning
        self.head = nn.Sequential(
            nn.Linear(head_hidden_dim * 4, head_hidden_dim),
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
        positive_coords = []
        negative_coords = []

        for mask in positive_masks:
            mask = np.array(mask, dtype=np.bool)
            mask = np.array(Image.fromarray(mask).resize(self.max_image_size))
            positive_pixels.append(embedded_img[mask])
            # Get spatial coordinates for positive pixels
            yy, xx = np.where(mask)
            positive_coords.append(torch.tensor(np.column_stack([xx, yy]), dtype=torch.float))

        for mask in negative_masks:
            mask = np.array(mask, dtype=np.bool)
            mask = np.array(Image.fromarray(mask).resize(self.max_image_size))
            negative_pixels.append(embedded_img[mask])
            # Get spatial coordinates for negative pixels
            yy, xx = np.where(mask)
            negative_coords.append(torch.tensor(np.column_stack([xx, yy]), dtype=torch.float))

        # Stack all positive and negative pixels and coordinates
        positive_pixels = torch.cat(positive_pixels, dim=0)
        positive_coords = torch.cat(positive_coords, dim=0) if positive_coords else torch.empty((0, 2))
        if negative_pixels:
            negative_pixels = torch.cat(negative_pixels, dim=0)
            negative_coords = torch.cat(negative_coords, dim=0) if negative_coords else torch.empty((0, 2))
        else:
            negative_pixels = torch.empty((0, embedded_img.shape[-1]), device=embedded_img.device)
            negative_coords = torch.empty((0, 2))

        # Combine positive and negative pixels and coordinates
        X = torch.cat([positive_pixels, negative_pixels], dim=0)
        spatial_coords = torch.cat([positive_coords, negative_coords], dim=0)

        # Create labels: 1 for positive, 0 for negative
        y = torch.cat(
            [
                torch.ones(positive_pixels.shape[0], 1),
                torch.zeros(negative_pixels.shape[0], 1),
            ],
            dim=0,
        )
        return X.to(self.device), y.to(self.device), spatial_coords.to(self.device)

    def _build_spatial_graph(self, embedded_img, spatial_coords):
        x = embedded_img  # Already flattened and filtered
        spatial_coords = spatial_coords / 511.0  # Normalize to [0, 1]

        edge_index = knn_graph(spatial_coords, k=3, batch=None, loop=True)

        return Data(x=x, edge_index=edge_index)

    def _train_head(self, X, y, spatial_coords):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            list(self.gat_layer.parameters()) + list(self.head.parameters()),
            lr=self.lr,
        )

        self.head.to(self.device)
        criterion.to(device=self.device)
        self.gat_layer.to(device=self.device)
        # Build spatial graph
        graph_data = self._build_spatial_graph(X, spatial_coords)
        graph_data = graph_data.to(self.device)
        y = y.to(self.device)

        # Train
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            gat_out = self.gat_layer(graph_data.x, graph_data.edge_index)
            outputs = self.head(gat_out)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

    def process_request(self, image, request: CompletionRequest):
        pbar = tqdm(desc="Processing request", total=5)
        pbar.set_postfix({"step": "Embedding image"})
        # Preprocess image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.resize(self.max_image_size)

        # Embed the image and standardize embeddings
        embedded_img = self.backbone.embed_image(image=image, standardize=True, keep_dim=True)
        print(embedded_img.shape)
        pbar.update(1)
        pbar.set_postfix({"step": "Preparing training data"})

        # Prepare training data from positive and negative masks
        X, y, spatial_coords = self._prepare_training_data(
            embedded_img,
            request.positive_exemplar_masks,
            request.negative_exemplar_masks
        )
        pbar.update(1)
        pbar.set_postfix({"step": "Training model."})

        # Train the head
        self._train_head(X, y, spatial_coords)

        pbar.update(1)
        pbar.set_postfix({"step": "Inferring similarity scores"})
        # Inference: score all pixels using the trained head
        with torch.no_grad():
            h, w = embedded_img.shape[0], embedded_img.shape[1]
            embedded_img_flat = embedded_img.reshape(-1, embedded_img.shape[-1])

            # Generate spatial coordinates for ALL pixels
            yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w))
            all_spatial_coords = torch.stack([xx.flatten(), yy.flatten()], dim=1).float().to(self.device)
            all_spatial_coords = all_spatial_coords / 511.0  # Normalize to [0, 1]

            # Build graph for ALL pixels
            inference_graph_data = self._build_spatial_graph(embedded_img_flat, all_spatial_coords)

            # Apply graph attention
            gat_out = self.gat_layer(inference_graph_data.x, inference_graph_data.edge_index)

            # Pass through MLP head
            scores = self.head(gat_out)
            scores = scores.reshape(h, w).cpu().numpy()

        debug_show_image(scores)

        # Convert scores to 0-255 range for thresholding
        scores = (scores * 255).astype(np.uint8)

        pbar.update(1)
        pbar.set_postfix({"step": "Postprocessing"})
        # Adaptive thresholding: use median of scores under positive masks
        combined_seed_mask = cv2.resize(request.combined_exemplar_mask, self.max_image_size)
        threshold = np.median(scores[combined_seed_mask]).item() - 1
        _, thresholded = cv2.threshold(scores, threshold, 255, cv2.THRESH_BINARY)
        # Extract masklets
        masklets, scores = extract_masklets(thresholded, scores)
        pbar.update(1)
        return masklets, scores
