# Use a lightweight Python base image with uv
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update --allow-unauthenticated && \
    apt-get install -y --no-install-recommends --allow-unauthenticated \
    git \
    openssh-client \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the files needed for dependency installation
COPY . .

# Add GitHub to known_hosts and sync dependencies using uv with SSH mount for private repos
RUN --mount=type=ssh mkdir -p ~/.ssh && \
    ssh-keyscan github.com >> ~/.ssh/known_hosts && \
    uv sync --no-cache

# Install torch (CPU version by default, can be customized for CUDA)
RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# Create necessary directories
RUN mkdir -p /app/logs /app/weights

# Expose the FastAPI port
EXPOSE 8003

# Run the FastAPI app using uvicorn
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
