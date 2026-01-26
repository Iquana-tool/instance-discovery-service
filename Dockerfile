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

# Configure git to use SSH for GitHub
RUN mkdir -p /root/.ssh && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts && \
    git config --global url."git@github.com:".insteadOf "https://github.com/"

# Copy only the files needed for dependency installation
COPY . .

# Setup SSH for private repo access
RUN mkdir -p ~/.ssh && \
    ssh-keyscan github.com >> ~/.ssh/known_hosts && \
    chmod 700 ~/.ssh && \
    chmod 644 ~/.ssh/known_hosts

# Copy SSH key if it exists (for build-time git access)
COPY build_key /tmp/build_key
RUN cp /tmp/build_key ~/.ssh/id_rsa && \
    chmod 600 ~/.ssh/id_rsa

# Sync dependencies using uv
RUN uv sync --no-cache

# Remove SSH key after installation
RUN rm -f ~/.ssh/id_rsa /tmp/build_key

# Install torch (CPU version by default, can be customized for CUDA)
RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# Create necessary directories
RUN mkdir -p /app/logs /app/weights

# Expose the FastAPI port
EXPOSE 8003

# Run the FastAPI app using uvicorn
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
