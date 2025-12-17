import threading
from collections import OrderedDict


class ModelCache:
    def __init__(self, size_limit=3):
        """
        Cache to hold a limited number of loaded models. Thread-safe.
        :param size_limit: Maximum number of models to cache. Default is 3.
        """
        self.size_limit = size_limit
        self.cache = OrderedDict()
        self.lock = threading.Lock()  # Lock for thread safety

    def get(self, user_identifier):
        with self.lock:  # Acquire lock
            if user_identifier in self.cache:
                entry = self.cache.pop(user_identifier)
                self.cache[user_identifier] = entry
                return entry[1]
            else:
                raise KeyError(user_identifier)

    def put(self, user_identifier, model_key, model):
        with self.lock:  # Acquire lock
            if user_identifier in self.cache:
                self.cache.pop(user_identifier)
            elif len(self.cache) >= self.size_limit:
                self.cache.popitem(last=False)
            self.cache[user_identifier] = (model_key, model)

    def check_if_loaded(self, user_identifier, model_key):
        with self.lock:  # Acquire lock
            return user_identifier in self.cache and self.cache[user_identifier][0] == model_key

    def __contains__(self, user_identifier):
        with self.lock: return user_identifier in self.cache
