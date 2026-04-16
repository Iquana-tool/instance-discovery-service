from os import getenv
import os

# General paths
ROOT = os.path.dirname(os.path.realpath(__file__))
LOG_DIR = getenv("LOG_DIR", "logs")
HF_ACCESS_TOKEN = getenv("HF_ACCESS_TOKEN")
REDIS_URL = getenv("REDIS_URL", "redis://localhost:6739")
MLFLOW_URL = getenv("MLFLOW_URL", "http://localhost:5000")
# Weights
WEIGHTS = os.path.join(ROOT, "weights")
DINO_PATH = os.path.join(WEIGHTS, "dino")
DINO_REPO_DIR = getenv("DINO_REPO_DIR")