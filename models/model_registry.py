from logging import getLogger
from schemas.models import CompletionModel as ModelInfo

logger = getLogger(__name__)


class ModelLoader:
    def __init__(self, loader_function, **kwargs):
        """
        Class to handle loading of models.
        :param loader_function: Function that loads the model.
        :param kwargs: Parameters to be passed to the loader function.
        """
        self.loader_function = loader_function
        self.kwargs = kwargs

    def is_loadable(self):
        # Implement logic to check if the model can be loaded with the given kwargs
        return True

    def load_model(self):
        return self.loader_function(**self.kwargs)


class ModelRegistry:
    def __init__(self):
        """Registry to hold and manage multiple models."""
        self.model_infos: dict[str, ModelInfo] = {}
        self.model_loaders: dict[str, ModelLoader] = {}

    def register_model(self,
                       model_info: ModelInfo,
                       model_loader: ModelLoader):
        """Register a new model in the registry.
        :param model_info: ModelInfo object.
        :param model_loader: ModelLoader object.
        :raises ValueError: If the model identifier is already registered.
        """
        if model_info.model_registry_key in self.model_infos:
            raise ValueError(f"Model with identifier {model_info.model_registry_key} is already registered.")
        if model_info.model_registry_key in self.model_loaders:
            raise ValueError(f"Model loader with identifier {model_info.model_registry_key} is already registered.")
        self.model_infos[model_info.model_registry_key] = model_info
        self.model_loaders[model_info.model_registry_key] = model_loader
        logger.info(f"Registered model {model_info.model_registry_key}. Model is loadable: {model_loader.is_loadable()}")

    def get_model_info(self, model_registry_key: str) -> ModelInfo:
        """Get the model information for the given identifier."""
        if model_registry_key not in self.model_infos:
            raise KeyError(f"Model with identifier {model_registry_key} is not registered.")
        return self.model_infos[model_registry_key]

    def get_model_loader(self, model_registry_key: str) -> ModelLoader:
        """Get the model loader for the given identifier."""
        if model_registry_key not in self.model_loaders:
            raise KeyError(f"Model loader with identifier {model_registry_key} is not registered.")
        return self.model_loaders[model_registry_key]

    def check_model_is_loadable(self, model_registry_key: str) -> bool:
        """Check if the model with the given identifier is loadable."""
        model = self.get_model_loader(model_registry_key)
        return model.is_loadable()

    def list_models(self, only_return_available: bool = True) -> list[ModelInfo]:
        """List all registered models.
        :param only_return_available: If True, only return models that are loadable. Default is True.
        :return: List of ModelInfo objects.
        """
        if only_return_available:
            # Only return loadable models
            return [model_info for model_info, model_loader in zip(self.model_infos.values(), self.model_loaders.values()) if model_loader.is_loadable()]
        return list(self.model_infos.values())

    def load_model(self, model_registry_key: str):
        """Load the model with the given identifier."""
        return self.get_model_loader(model_registry_key).load_model()
