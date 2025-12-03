from enum import Enum

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from transformers import pipeline, AutoModel, AutoImageProcessor

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


class DinoModel(Encoder):
    def __init__(self,
                 model_type: DinoModelType,
                 patch_size,
                 image_size,
                 preprocess_mean=(0.485, 0.456, 0.406),
                 preprocess_std=(0.229, 0.224, 0.225),
                 device='cuda'):
        self.model_type = model_type
        hf_url = MODEL_TO_HF_URL[model_type]
        self.processor = AutoImageProcessor.from_pretrained(hf_url)
        self.model = AutoModel.from_pretrained(
            hf_url,
            device_map="auto",
        )
        self.device = device
        self.n_layers = MODEL_TO_NUM_LAYERS[model_type]
        self.patch_size = patch_size
        self.image_size = image_size
        self.preprocess_mean = preprocess_mean
        self.preprocess_std = preprocess_std

    def preprocess(self, image: Image.Image):
        w = image.width
        h = image.height
        h_patches = int(self.image_size / self.patch_size)
        w_patches = int((w * self.image_size) / (h * self.patch_size))
        tensor = self.processor(image=image, return_tensors="pt")
        return tensor, h_patches, w_patches

    def embed_preprocessed(self, tensor: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                feats = self.model(tensor.unsqueeze(0).cuda(),
                                   n=range(self.n_layers),
                                   reshape=True,
                                   norm=True)
                x = feats[-1].squeeze().detach().cpu()
                dim = x.shape[0]
                return x.view(dim, -1).permute(1, 0)

    def embed_image(self, image: Image.Image, keep_dim=True) -> torch.Tensor:
        og_h, og_w = image.height, image.width
        tensor, h_patches, w_patches = self.preprocess(image)
        flat_embedding = self.embed_preprocessed(tensor)
        n_features = flat_embedding.shape[1]
        reshaped_embedding = flat_embedding.reshape(h_patches, w_patches, n_features)
        if keep_dim:
            reshaped_embedding = TF.resize(
                reshaped_embedding.permute(2, 0, 1),
                [og_h, og_w]
            ).permute(1, 2, 0)
        return reshaped_embedding
