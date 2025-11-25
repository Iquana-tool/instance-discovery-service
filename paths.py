from os import getenv
import os

# General paths
ROOT = os.path.dirname(os.path.realpath(__file__))
LOG_DIR = getenv("LOG_DIR", "logs")