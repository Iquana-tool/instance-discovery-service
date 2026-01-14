import numpy as np
from PIL.Image import fromarray
from scipy.ndimage import binary_fill_holes
from sklearn.cluster import DBSCAN

from app.schemas.inference import InstanceMasksResponse
from models.base_models import BaseModel
from models.encoders.dino_encoder import DinoModel, DinoModelType
from util.postprocess import extract_masklets


class DBSCANPredictor:
    def __init__(self, eps=0.5, min_samples=5, metric='cosine'):
        """
        Initialize DBSCAN predictor.

        Args:
            eps: Maximum distance between two samples for one to be considered as in the neighborhood of the other.
            min_samples: Number of samples in a neighborhood for a point to be considered a core point.
            metric: Distance metric for DBSCAN (e.g., 'cosine', 'euclidean').
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.dbscan = None
        self.labels_ = None

    def fit(self, embeddings):
        """
        Fit DBSCAN to embeddings.

        Args:
            embeddings: Tensor of shape (N, D) where N is the number of pixels and D is the embedding dimension.
        """
        # Convert to numpy and normalize
        embeddings_np = embeddings.cpu().numpy()
        embeddings_np = embeddings_np / np.linalg.norm(embeddings_np, axis=1, keepdims=True)

        # Fit DBSCAN
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric)
        self.labels_ = self.dbscan.fit_predict(embeddings_np)

        return self.labels_

    def predict(self, embeddings, seed_labels):
        """
        Predict clusters and return a mask for the cluster containing the majority of seed pixels.

        Args:
            embeddings: Tensor of shape (N, D).
            seed_labels: Binary mask indicating seed pixels (e.g., exemplars).

        Returns:
            Binary mask of shape (H, W) indicating pixels assigned to the same cluster as the seeds.
        """
        # Fit DBSCAN
        labels = self.fit(embeddings)

        # Reshape embeddings and labels to 2D
        H, W, D = embeddings.shape
        labels_2d = labels.reshape(H, W)
        seed_labels_flat = seed_labels.reshape(-1)

        # Find the most common cluster label among seed pixels
        seed_cluster_labels = labels_2d[seed_labels_flat]
        if len(seed_cluster_labels) == 0:
            return np.zeros((H, W), dtype=bool)

        # Get the most frequent cluster label among seeds
        target_cluster = np.bincount(seed_cluster_labels[seed_cluster_labels != -1]).argmax()
        target_cluster_mask = (labels_2d == target_cluster)

        # Fill holes in the mask for better segmentation
        target_cluster_mask = binary_fill_holes(target_cluster_mask)

        return target_cluster_mask


# Example usage in your `CosineSimilarityModel`:
class DBScanModel(BaseModel):
    def __init__(self, max_image_size=512):
        self.predictor = DBSCANPredictor()
        self.backbone = DinoModel(
            device="auto",
            model_type=DinoModelType.VITS16,
            patch_size=16,
            image_size=1024,
        )
        self.max_image_size = max_image_size if isinstance(max_image_size, list) else [max_image_size, max_image_size]

    def process_request(self, image, request):
        # 1. Preprocess image and embed
        if isinstance(image, np.ndarray):
            image = fromarray(image)
        image = image.resize(self.max_image_size)
        embedded_img = self.backbone.embed_image(image=image)

        # 2. Combine seed masks
        combined_seed_mask = request.get_combined_seed_mask(self.max_image_size)

        # 3. Flatten embeddings and seed mask
        H, W = combined_seed_mask.shape
        embeddings_flat = embedded_img.reshape(-1, embedded_img.shape[-1])
        seed_mask_flat = combined_seed_mask.reshape(-1).astype(bool)

        # 4. Predict clusters using DBSCAN
        dbscan_predictor = DBSCANPredictor(eps=0.3, min_samples=10, metric='cosine')
        cluster_mask = dbscan_predictor.predict(embeddings_flat, seed_mask_flat)

        # 5. Post-process cluster mask (e.g., connected components)
        masklets, scores = extract_masklets(cluster_mask)

        return InstanceMasksResponse.from_masks(masklets, scores, request)
