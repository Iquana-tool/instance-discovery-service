import os
from contextlib import asynccontextmanager
from logging import DEBUG
from logging import getLogger

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import login, logout, whoami
from app.state import MODEL_CACHE, MODEL_REGISTRY
from models.register_models import register_models
from paths import HF_ACCESS_TOKEN

# Router imports
from app.routes import router as health_router
from app.routes.models import router as model_router
from app.routes.models import session_router as model_session_router
from app.routes.images import router as image_router
from app.routes.images import session_router as image_session_router
from app.routes.inference import router as inference_router
from app.routes.inference import session_router as inference_session_router

logger = getLogger(__name__)
logger.setLevel(DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    logger.debug("Starting up the Prompted Segmentation Service")
    logger.debug("Registering models in the MODEL_REGISTRY")
    login(token=HF_ACCESS_TOKEN)
    try:
        user_info = whoami()
        print("You are logged in as:", user_info["name"])
    except Exception as e:
        print("You are not logged in. Error:", e)
    register_models(MODEL_REGISTRY)
    yield
    # Shutdown code
    logout()
    logger.debug("Shutting down the Prompted Segmentation Service")


def create_app():
    logger.debug("Creating FastAPI application")
    # Load environment variables
    load_dotenv()

    # Get allowed origins from environment variable
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000").split(",")

    app = FastAPI(
        title="Completion Segmentation API",
        lifespan=lifespan,
        description="FastAPI backend for interactive completion segmentation",
        version="0.1.0",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include the routers
    app.include_router(health_router)
    app.include_router(image_router)
    app.include_router(image_session_router)
    app.include_router(inference_router)
    app.include_router(inference_session_router)
    app.include_router(model_router)
    app.include_router(model_session_router)

    return app