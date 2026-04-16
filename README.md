# IQUANA Instance Discovery Service

FastAPI service for instance discovery and prompted/few-shot segmentation.

It exposes endpoints to:
- inspect service health,
- list and inspect registered models,
- preload models for an annotation session,
- run inference from a completion request payload.

## What this service does

Given an input image plus prompt/exemplar information (via `CompletionRequest`),
the service loads the selected model from an MLflow-backed registry and returns
detected object contours.

## Architecture at a glance

- App entrypoint: `main.py`
- FastAPI app factory and startup lifecycle: `app/__init__.py`
- Shared model registry instance: `app/state.py`
- Model registration config: `models/register_models.py`
- API routes:
  - `app/routes/__init__.py` (`/health`)
  - `app/routes/models.py` (`/models/*` and preload route)
  - `app/routes/inference.py` (`/annotation_session/run`)

Startup flow:
1. Load environment variables (`dotenv`).
2. Optionally authenticate to Hugging Face using `HF_ACCESS_TOKEN`.
3. Register configured models in MLflow if they are not already registered.
4. Serve FastAPI routes.

## Requirements

- Python `>=3.12`
- `uv` package manager (recommended, used by this repo)
- Access to an MLflow tracking/registry server (default `http://localhost:5000`)
- Hugging Face token if model downloads require authentication

## Quick start (local)

1. Create a `.env` file from `env.example`.
2. Set at least `HF_ACCESS_TOKEN` if needed for model access.
3. Install dependencies.
4. Run the API.

```powershell
Copy-Item env.example .env
uv sync
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv run uvicorn main:app --host 0.0.0.0 --port 8003 --reload
```

API docs will be available at:
- `http://localhost:8003/docs`
- `http://localhost:8003/redoc`

## Run with Docker

The repository includes `Dockerfile` and `docker-compose.yml`.

```powershell
docker compose up --build
```

Service endpoint:
- `http://localhost:8003`

## Environment variables

From `env.example` and `paths.py`:

| Variable | Default | Purpose |
|---|---|---|
| `HF_ACCESS_TOKEN` | unset | Hugging Face auth token used at startup (`login`). |
| `ALLOWED_ORIGINS` | `http://localhost:8000` | Comma-separated CORS origins. |
| `PORT` | `8003` | Declared in env/compose; current run commands still bind explicitly to `8003`. |
| `LOG_DIR` | `logs` | Directory for `logs/logs.txt`. |
| `MLFLOW_URL` | `http://localhost:5000` | MLflow registry/tracking URL for model registry backend. |
| `REDIS_URL` | `redis://localhost:6739` | Celery broker/backend base URL (`celery_app.py`). |
| `DINO_REPO_DIR` | unset | Optional path used by DINO-related components. |

## API endpoints

### `GET /health`
Returns runtime health and device information.

Example response fields:
- `status`
- `device`
- `torch_version`

### `GET /models/all`
Lists models in registry filtered by tags:
- `task=instance-discovery`

### `GET /models/all/available`
Lists models filtered by tags:
- `task=instance-discovery`
- `status=ready`

### `GET /models/{model_registry_key}`
Fetches detailed model metadata by registry key.

### `GET /annotation_session/models/{model_registry_key}/preload?user_id=<id>`
Preloads the selected model alias (`latest`) into cache.

### `POST /annotation_session/run`
Runs inference for a `CompletionRequest` payload and returns contour results.

Note: `CompletionRequest` schema comes from `iquana_toolbox`:
`iquana_toolbox.schemas.networking.http.services.CompletionRequest`.

## Registered models

Configured in `models/register_models.py`:
- `sam3` (ready)
- `sansa` (ready)
- `geco` (ready)
- `few-shot-attention` (experimental)
- `watershed-dino` (experimental)

At startup, each model is registered only if it is not already present in MLflow.

## Troubleshooting

- If startup logs show Hugging Face login issues, verify `HF_ACCESS_TOKEN`.
- If model listing is empty, verify MLflow connectivity (`MLFLOW_URL`) and registration logs.
- If `GET /models/all` does not include a model, check its tags in MLflow. This endpoint filters by `task=instance-discovery`.
- If inference fails, ensure the requested `model_registry_key` exists and has a `latest` alias.

## Development notes

- Logging is configured in `main.py` and writes to `logs/logs.txt` (or `LOG_DIR`).
- Celery app definition exists in `celery_app.py` but no worker commands are documented in this repository yet.
