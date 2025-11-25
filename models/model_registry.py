from logging import getLogger


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
        return False

    def load_model(self):
        return self.loader_function(**self.kwargs)


class ModelInfo:
    def __init__(self,
                 identifier_str: str,
                 name: str,
                 description: str,
                 tags: list[str],
                 supported_prompt_types: list[str] = ["point", "box"],
                 supports_refinement: bool = False):
        """Class to hold information about a segmentation model.
        :param identifier_str: Identifier string. Must be unique.
        :param name: Human readable name of the model.
        :param description: Description of the model.
        :param tags: List of tags for the model.
        :param supported_prompt_types: List of supported prompt types. Default is ["point", "box"].
        :param supports_refinement: Whether the model supports refinement with a previous mask. Default is False.
        :raises ValueError: If the identifier string is invalid.
        """
        self.identifier_str = identifier_str
        self.name = name
        self.description = description
        self.tags = tags
        self.supported_prompt_types = supported_prompt_types
        self.supports_refinement = supports_refinement

    def to_json(self):
        """Convert the model information to a JSON-serializable dictionary."""
        return {
            "identifier_str": self.identifier_str,
            "name": self.name,
            "description": self.description,
            "weights_path": self.weights_path,
            "configs_path": self.configs_path,
            "tags": self.tags,
            "supported_prompt_types": self.supported_prompt_types,
            "supports_refinement": self.supports_refinement
        }


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
        if model_info.identifier_str in self.model_infos:
            raise ValueError(f"Model with identifier {model_info.identifier_str} is already registered.")
        if model_info.identifier_str in self.model_loaders:
            raise ValueError(f"Model loader with identifier {model_info.identifier_str} is already registered.")
        self.model_infos[model_info.identifier_str] = model_info
        self.model_loaders[model_info.identifier_str] = model_loader
        logger.info(f"Registered model {model_info.identifier_str}. Model is loadable: {model_loader.is_loadable()}")

    def get_model_info(self, identifier_str: str) -> ModelInfo:
        """Get the model information for the given identifier."""
        if identifier_str not in self.model_infos:
            raise KeyError(f"Model with identifier {identifier_str} is not registered.")
        return self.model_infos[identifier_str]

    def get_model_loader(self, identifier_str: str) -> ModelLoader:
        """Get the model loader for the given identifier."""
        if identifier_str not in self.model_loaders:
            raise KeyError(f"Model loader with identifier {identifier_str} is not registered.")
        return self.model_loaders[identifier_str]

    def check_model_is_loadable(self, identifier_str: str) -> bool:
        """Check if the model with the given identifier is loadable."""
        model = self.get_model_loader(identifier_str)
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

    def load_model(self, identifier_str: str):
        """Load the model with the given identifier."""
        return self.get_model_loader(identifier_str).load_model()
