from enum import Enum
from typing import Literal

import plotly.express as px
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.decomposition import PCA
from transformers import AutoImageProcessor, DINOv3ViTModel, DINOv3ViTConfig, DINOv3ViTImageProcessorFast

from models.encoders.encoder import Encoder


class DinoModelType(Enum):
    VITS16 = "dinov3_vits16"
    VITS16PLUS = "dinov3_vits16plus"
    VITB16 = "dinov3_vitb16"
    VITL16 = "dinov3_vitl16"
    VITH16PLUS = "dinov3_vith16plus"
    VIT7B16 = "dinov3_vit7b16"

MODEL_TO_WEIGHTS= {
    DinoModelType.VITS16: None,
    DinoModelType.VITS16PLUS: "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
    DinoModelType.VITB16: None,
    DinoModelType.VITL16: None,
    DinoModelType.VITH16PLUS: None,
    DinoModelType.VIT7B16: None,
}
MODEL_TO_HF_URL = {
    DinoModelType.VITS16: "facebook/dinov3-vits16-pretrain-lvd1689m",
    DinoModelType.VITS16PLUS: "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    DinoModelType.VITB16: "facebook/dinov3-vitb16-pretrain-lvd1689m",
    DinoModelType.VITL16: "facebook/dinov3-vitl16-pretrain-lvd1689m",
    DinoModelType.VITH16PLUS: "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    DinoModelType.VIT7B16: "facebook/dinov3-vit7b16-pretrain-lvd1689m",
}
MODEL_TO_NUM_LAYERS = {
    DinoModelType.VITS16: 12,
    DinoModelType.VITS16PLUS: 12,
    DinoModelType.VITB16: 12,
    DinoModelType.VITL16: 24,
    DinoModelType.VITH16PLUS: 32,
    DinoModelType.VIT7B16: 40,
}


def uniform_sphere_energy(embeddings, lr=0.01, steps=100, beta=1.0):
    """
    Redistribute embeddings on a hypersphere to minimize energy (maximize uniformity).

    Args:
        embeddings: Tensor of shape (n_points, n_features)
        lr: Learning rate for gradient descent
        steps: Number of optimization steps
        beta: Strength of repulsion (higher = stronger repulsion)
    """
    # Ensure embeddings are on the unit sphere
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Clone to avoid modifying the original tensor
    embeddings = embeddings.clone().detach().requires_grad_(True)

    # Optimizer
    optimizer = torch.optim.Adam([embeddings], lr=lr)

    # Energy function: sum of inverse distances
    def energy(x):
        # Pairwise distances (cosine distance on the sphere)
        dists = 1 - torch.mm(x, x.t())  # 1 - cos(theta) = 2*sin^2(theta/2)
        # Avoid division by zero and self-distance
        dists = dists * (1 - torch.eye(dists.shape[0], device=x.device))
        # Inverse distance energy
        E = torch.sum(1.0 / (dists + 1e-8))
        return E

    # Optimization loop
    for _ in range(steps):
        optimizer.zero_grad()
        E = energy(embeddings)
        E.backward()
        optimizer.step()
        # Re-project to the unit sphere after each step
        with torch.no_grad():
            embeddings.data = F.normalize(embeddings, p=2, dim=1)

    return embeddings.detach()


class DinoModel(Encoder):
    def __init__(self,
                 model_type: DinoModelType,
                 patch_size=None,
                 image_size=None,
                 preprocess_mean=(0.485, 0.456, 0.406),
                 preprocess_std=(0.229, 0.224, 0.225),
                 device='auto'):
        self.model_type = model_type
        hf_url = MODEL_TO_HF_URL[model_type]
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device

        self.n_layers = MODEL_TO_NUM_LAYERS[model_type]
        self.patch_size = patch_size
        self.image_size = image_size
        self.preprocess_mean = preprocess_mean
        self.preprocess_std = preprocess_std

        self.processor: DINOv3ViTImageProcessorFast = AutoImageProcessor.from_pretrained(
            hf_url,
            device=self.device)
        self.config = DINOv3ViTConfig.from_pretrained(hf_url, device=self.device)
        if self.patch_size is not None:
            self.config.patch_size = self.patch_size
        else:
            self.patch_size = self.config.patch_size
        if self.image_size is not None:
            self.config.image_size = self.image_size
        else:
            self.image_size = self.config.image_size
        self.model = DINOv3ViTModel(
            config=self.config
        )
        self.model.to(self.device)

    def preprocess(self, image: Image.Image):
        w = image.width
        h = image.height
        h_patches = int(self.image_size / self.patch_size)
        w_patches = int((w * self.image_size) / (h * self.patch_size))
        tensor = self.processor(
            images=image,
            size=self.image_size,
            return_tensors="pt")
        return tensor, h_patches, w_patches

    def embed_preprocessed(self, input) -> torch.Tensor:
        with torch.inference_mode():
            with torch.autocast(device_type=self.device, dtype=torch.float32):
                feats = self.model(**input)
                x = feats[-1].squeeze().detach().cpu()
                dim = x.shape[0]
                return x.view(dim, -1).permute(1, 0)

    def embed_image(self,
                    image: Image.Image,
                    standardize: bool = True,
                    reduced_features: int | None = None,
                    reduce_method: Literal["pca"] = "pca",
                    keep_dim: bool = True,
                    debug_pca: bool = False) -> torch.Tensor:
        with torch.inference_mode():
            with torch.autocast(device_type=self.device, dtype=torch.float32):
                # Save the original size
                og_h, og_w = image.height, image.width

                # Preprocess the image
                inputs, h_patches, w_patches = self.preprocess(image)
                print(h_patches, w_patches)

                # Compute the embedding
                with torch.inference_mode():
                    outputs = self.model.forward(**inputs, output_hidden_states=False)

                # We only need the last hidden state
                last_hidden_state = outputs.last_hidden_state.squeeze()
                cls_token, reg_token, embeddings = last_hidden_state[0], last_hidden_state[1:5], last_hidden_state[5:]

                if standardize:
                    # Standardize each feature dimension to zero mean and unit variance
                    mean = embeddings.mean(dim=0, keepdim=True)
                    std = embeddings.std(dim=0, keepdim=True)
                    embeddings = (embeddings - mean) / (std + 1e-8)  # Add small epsilon to avoid division by zero

                if reduced_features is not None and not reduced_features >= embeddings.shape[1]:
                    if reduce_method == "pca":
                        # Apply PCA and whitening
                        pca = PCA(n_components=reduced_features, whiten=True)
                        embeddings_cpu = embeddings.cpu().numpy()
                        embeddings_cpu = pca.fit_transform(embeddings_cpu)
                        embeddings = torch.tensor(embeddings_cpu, device=self.device)
                    else:
                        raise NotImplementedError(f"Feature reduction not implemented for method '{reduce_method}'")

                embeddings = embeddings.reshape(h_patches, w_patches, -1)
                # Resize to original size if enabled
                if keep_dim:
                    embeddings = TF.resize(
                        embeddings.permute(2, 0, 1),
                        [og_h, og_w]
                    ).permute(1, 2, 0)

                # Visualizing the PCA here for debugging
                if debug_pca:
                    pca = PCA(n_components=3, whiten=True)
                    output_cpu = embeddings.flatten(end_dim=1).cpu().numpy()
                    projection = pca.fit_transform(output_cpu).reshape(og_h, og_w, 3)
                    projection = (projection - projection.min()) / (projection.max() - projection.min())
                    px.imshow(projection).show()
                return embeddings

