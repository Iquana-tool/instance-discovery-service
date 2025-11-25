import cv2
import torch
import numpy as np

class CosineSimilarityPredictor:
    def __init__(self,
                 device: str = "cuda",
                 memory_aggregation: str = "mean",
                 similarity_aggregation: str = "max",
                 similarity_redistribution_method: str = "none"):
        self.example_vectors_list = []
        self.n_features = None
        self.device = device
        self.similarity_aggregation = similarity_aggregation
        self.memory_aggregation = memory_aggregation
        self.similarity_redistribution_method = similarity_redistribution_method

    @property
    def params(self):
        return {"memory_aggregation": self.memory_aggregation,
                "similarity_aggregation": self.similarity_aggregation,
                "similarity_redistribution_method": self.similarity_redistribution_method}

    @property
    def num_params(self):
        return len(self.example_vectors_list) * (self.n_features if self.n_features is not None else 0)

    @property
    def example_vectors(self):
        if not self.example_vectors_list:
            return torch.empty((0, self.n_features), device=self.device)
        return torch.stack(self.example_vectors_list).half().to(self.device)

    def reset_features(self):
        self.example_vectors_list = []

    @property
    def threshold(self):
        """ Dynamically compute the similarity threshold. It is defined as the minimum similarity required for all
            example vectors to be true.
        """
        if not self.example_vectors_list:
            return 0
        return torch.min(self.cosine_similarity(self.example_vectors)).item()

    @classmethod
    def cosine_similarity(cls, mat_1, mat_2=None, normalize=True):
        if mat_2 is None:
            mat_2 = mat_1
        assert mat_1.shape[1] == mat_2.shape[1], "Matrices must have same number of features."
        # Normalize
        if normalize:
            mat_1 = torch.nn.functional.normalize(mat_1, dim=1)
            mat_2 = torch.nn.functional.normalize(mat_2, dim=1)
        # Compute cosine similarities
        return torch.mm(mat_1, mat_2.T)

    def aggregate_similarities(self, mat: torch.Tensor):
        if self.similarity_aggregation == "max":
            return torch.max(mat, dim=1).values
        elif self.similarity_aggregation == "mean":
            return torch.mean(mat, dim=1)
        elif self.similarity_aggregation == "min":
            return torch.min(mat, dim=1).values
        else:
            raise NotImplementedError

    def add_example_vector(self, example_vector):
        """Add an exemplar feature vector to the predictor."""
        if isinstance(example_vector, np.ndarray):
            example_vector = torch.from_numpy(example_vector).half().to(self.device)
        assert len(example_vector.shape) == 1, f"Example vector must be 1-dimensional. You provided shape {example_vector.shape}."
        if self.n_features is None:
            self.n_features = example_vector.shape[0]
        else:
            assert example_vector.shape[0] == self.n_features, \
                f"Example vector must have {self.n_features} features. You provided {example_vector.shape[0]} features."
        self.example_vectors_list.append(example_vector)

    def add_example_vectors(self, example_vectors: torch.Tensor):
        """Add multiple exemplar feature vectors to the predictor."""
        if isinstance(example_vectors, np.ndarray):
            example_vectors = torch.from_numpy(example_vectors).half().to(self.device)
        if self.memory_aggregation == "none":
            for vec in example_vectors:
                self.add_example_vector(vec)
        elif self.memory_aggregation == "mean":
            self.add_example_vector(example_vectors.mean(dim=0))
        elif self.memory_aggregation == "max":
            self.add_example_vector(example_vectors.max(dim=0).values)
        elif self.memory_aggregation == "min":
            self.add_example_vector(example_vectors.min(dim=0).values)
        elif self.memory_aggregation == "none":
            for vec in example_vectors:
                self.add_example_vector(vec)
        else:
            raise NotImplementedError

    def predict_flat(self, image_vectors: torch.Tensor):
        """Predict if image embedding vectors are similar to any exemplar vector."""
        return self.get_flat_similarities(image_vectors) >= self.threshold

    def predict_mask(self, image_embedding: torch.Tensor):
        sim_map = self.get_similarity_map(image_embedding)
        return sim_map >= self.threshold

    def get_flat_similarities(self, image_vectors: torch.Tensor):
        assert image_vectors.shape[-1] == self.n_features, \
            f"Image embedding vectors must have {self.n_features} features. You provided {image_vectors.shape}."
        image_vectors = image_vectors.half().to(self.device)
        # Compute cosine similarities
        cosine_similarities = self.cosine_similarity(image_vectors, self.example_vectors)
        # Aggregate features
        return self.aggregate_similarities(cosine_similarities)

    def get_similarity_map(self, image_embedding: torch.Tensor):
        if isinstance(image_embedding, np.ndarray):
            image_embedding = torch.from_numpy(image_embedding)
        image_embedding = image_embedding.half().to(self.device)
        og_shape = image_embedding.shape
        image_embedding_vectors = image_embedding.reshape(-1, self.n_features)
        flat_map = self.get_flat_similarities(image_embedding_vectors)
        return self.redistribute_similarities((flat_map.reshape(og_shape[:-1]) + 1) / 2)

    def redistribute_similarities(self, sim_map: torch.Tensor):
        if self.similarity_redistribution_method == "none":
            return sim_map
        else:
            sim_map_np = sim_map.cpu().numpy()
            if self.similarity_redistribution_method == "norm":
                return_sim_map = (sim_map_np - np.min(sim_map_np)) / (np.max(sim_map_np) - np.min(sim_map_np))
            elif self.similarity_redistribution_method == "he":
                return_sim_map = cv2.equalizeHist((sim_map_np * 255).astype(np.uint8))
                return_sim_map = (return_sim_map / 255).astype(float)
            elif self.similarity_redistribution_method == "clahe":
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                return_sim_map = clahe.apply((sim_map_np * 255).astype(np.uint8))
                return_sim_map = (return_sim_map / 255).astype(float)
            else:
                raise NotImplementedError
            return torch.from_numpy(return_sim_map)