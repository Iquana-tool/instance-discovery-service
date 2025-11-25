from os import getenv
import os

# General paths
ROOT = os.path.dirname(os.path.realpath(__file__))
LOG_DIR = getenv("LOG_DIR", "logs")


# Weights
WEIGHTS = os.path.join(ROOT, "weights")
DINO_PATH = os.path.join(WEIGHTS, "dino")