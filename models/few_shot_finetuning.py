import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from typing import Union, Literal
from tqdm import tqdm

from iquana_toolbox.schemas.contours import Contour
from iquana_toolbox.schemas.service_requests import CompletionRequest
from models.base_models import BaseModel
from models.encoders.dino_encoder import DinoModel, DinoModelType
from models.encoders.encoder_base_class import Encoder
from util.debug import debug_show_image
from util.misc import get_device_from_str
from util.postprocess import extract_masklets


class FocalLoss(nn.Module):
    """
    Focuses learning on hard examples by down-weighting easy background pixels.
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = bce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean()


class AttentionFewShotModel(BaseModel):
    """
    A robust few-shot instance discovery model using Cross-Attention,
    Spatial Positional Encodings, and Focal Loss.
    """

    def __init__(
            self,
            backbone: Encoder | None = None,
            max_image_size: Union[int, list[int]] = 256,
            head_hidden_dim: int = 128,
            num_epochs: int = 50,
            lr: float = 0.0005,
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

        self.max_image_size = [max_image_size, max_image_size] if isinstance(max_image_size, int) else max_image_size
        self.embedding_dim = self.backbone.embedding_dim

        # --- Spatial Positional Encoding ---
        # Maps [x, y] normalized coordinates to the feature space
        self.pos_emb = nn.Sequential(
            nn.Linear(2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, self.embedding_dim)
        )

        # --- Multi-Head Cross-Attention ---
        self.attn = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=8,
            batch_first=True
        )

        # --- Final Classifier ---
        self.head = nn.Sequential(
            nn.Linear(self.embedding_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, 1)
        )

        self.num_epochs = num_epochs
        self.lr = lr

    def _get_coords(self, h, w):
        """Generates a grid of normalized [x, y] coordinates."""
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, h, device=self.device),
            torch.linspace(0, 1, w, device=self.device),
            indexing='ij'
        )
        return torch.stack([xx, yy], dim=-1)  # [H, W, 2]

    def _get_patch_features(self, embedded_img, masks):
        """Extracts DINO features and spatial coordinates under masks."""
        if not masks:
            return torch.empty((0, self.embedding_dim), device=self.device), torch.empty((0, 2), device=self.device)

        h, w, c = embedded_img.shape
        coords_grid = self._get_coords(h, w)

        features, coords = [], []
        for mask in masks:
            mask_np = np.array(mask, dtype=np.bool_)
            mask_resized = cv2.resize(mask_np.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            mask_bool = torch.from_numpy(mask_resized).bool().to(self.device)

            features.append(embedded_img[mask_bool])
            coords.append(coords_grid[mask_bool])

        return torch.cat(features), torch.cat(coords)

    def _train_head(self, features, coords, labels):
        """Fine-tunes the attention and MLP head on the provided exemplars."""
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        criterion.to(self.device)
        optimizer = optim.Adam(
            list(self.attn.parameters()) +
            list(self.head.parameters()) +
            list(self.pos_emb.parameters()),
            lr=self.lr
        )
        # Device Preparation
        self.attn.to(self.device)
        self.pos_emb.to(self.device)
        self.head.to(self.device)
        criterion.to(self.device)

        self.attn.train()
        self.head.train()
        self.pos_emb.train()


        for epoch in range(self.num_epochs):
            optimizer.zero_grad()

            # Inject spatial position into semantic features
            spatial_info = self.pos_emb(coords)
            x_rich = (features + spatial_info).unsqueeze(0)  # [1, N_exemplars, Dim]

            # Self-attention allows exemplars to contextualize each other
            attn_out, _ = self.attn(x_rich, x_rich, x_rich)
            logits = self.head(attn_out.squeeze(0))
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    def process_request(self, image, request: CompletionRequest):
        pbar = tqdm(desc="Processing Request", total=5)

        # 1. Image Embedding
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.resize(self.max_image_size)
        embedded_img = self.backbone.embed_image(image=image, standardize=True, keep_dim=True)
        h_f, w_f, c_f = embedded_img.shape
        pbar.update(1)

        # 2. Data Preparation
        pos_f, pos_c = self._get_patch_features(embedded_img, request.positive_exemplar_masks)
        neg_f, neg_c = self._get_patch_features(embedded_img, request.negative_exemplar_masks)

        X_feats = torch.cat([pos_f, neg_f])
        X_coords = torch.cat([pos_c, neg_c])
        X_feats.to(self.device)
        X_coords.to(self.device)
        y_train = torch.cat([
            torch.ones(pos_f.size(0), 1),
            torch.zeros(neg_f.size(0), 1)
        ]).to(self.device)
        pbar.update(1)

        # 3. Training
        self._train_head(X_feats, X_coords, y_train)
        pbar.update(1)

        # 4. Global Inference
        self.attn.eval()
        self.head.eval()
        self.pos_emb.eval()
        with torch.no_grad():
            img_coords = self._get_coords(h_f, w_f).reshape(-1, 2)
            img_coords.to(self.device)
            img_flat = embedded_img.reshape(1, -1, c_f)

            # Combine Image + Position
            query_rich = img_flat + self.pos_emb(img_coords).unsqueeze(0)

            # Combine Exemplars + Position
            bank_rich = (X_feats + self.pos_emb(X_coords)).unsqueeze(0)

            # Cross-Attention: Every patch compares itself to the memory bank
            context_feats, _ = self.attn(query_rich, bank_rich, bank_rich)
            logits = self.head(context_feats + query_rich)  # Residual link
            scores = torch.sigmoid(logits).reshape(h_f, w_f).cpu().numpy()

        pbar.update(1)

        # 5. Post-processing
        scores_uint8 = (scores * 255).astype(np.uint8)
        combined_mask = cv2.resize(request.combined_exemplar_mask.astype(np.uint8),
                                   (w_f, h_f), interpolation=cv2.INTER_NEAREST)

        # Determine threshold based on exemplar performance
        threshold = np.median(scores_uint8[combined_mask > 0]).item() if np.any(combined_mask) else 127
        _, thresholded = cv2.threshold(scores_uint8, int(threshold) - 1, 255, cv2.THRESH_BINARY)

        # Upscale to original request size
        thresholded = cv2.resize(thresholded, self.max_image_size, interpolation=cv2.INTER_NEAREST)
        scores_full = cv2.resize(scores_uint8, self.max_image_size, interpolation=cv2.INTER_LINEAR)

        masklets, final_scores = extract_masklets(thresholded, scores_full)
        pbar.update(1)

        return masklets, final_scores
