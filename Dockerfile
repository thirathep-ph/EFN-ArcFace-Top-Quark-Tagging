# Base image with PyTorch and CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirement files and install
COPY setup.py README.md ./
# We install dependencies via setup.py
RUN pip install --no-cache-dir -e .

# Copy the rest of the application
COPY jet_tagging/ ./jet_tagging/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Default command: show help for the training script
CMD ["python", "scripts/train_top.py", "--help"]
