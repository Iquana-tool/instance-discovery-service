# syntax=docker/dockerfile:1
# Use a lightweight Python base image with uv
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set the working directory
WORKDIR /app

# Install system dependencies including openssh-client for git SSH
RUN apt-get update --allow-unauthenticated && \
    apt-get install -y --no-install-recommends --allow-unauthenticated \
    git \
    openssh-client \
    libgl1 \
    libglib2.0-0 \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Configure git to use token for private repo access (before copying code)
ARG GITHUB_TOKEN
RUN if [ -n "$GITHUB_TOKEN" ]; then \
    cd /tmp && \
    echo "https://${GITHUB_TOKEN}@github.com" > /root/.git-credentials && \
    GIT_CONFIG_GLOBAL=/root/.gitconfig git config --global credential.helper store && \
    GIT_CONFIG_GLOBAL=/root/.gitconfig git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"; \
    fi

# Copy only the files needed for dependency installation
COPY . .

# Sync dependencies using uv
RUN uv sync --no-cache

# Install torch (CPU version by default, can be customized for CUDA)
RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# Create necessary directories
RUN mkdir -p /app/logs /app/weights

# Expose the FastAPI port
EXPOSE 8003

# Run the FastAPI app using uvicorn
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
