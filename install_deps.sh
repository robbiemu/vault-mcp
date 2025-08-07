#!/bin/bash

# This script installs all project dependencies, including dev dependencies,
# ensuring that CUDA-related packages are pinned to their CPU-only versions.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Creating a virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing core dependencies..."
pip install \
    "fastapi>=0.104.0" \
    "uvicorn[standard]>=0.24.0" \
    "watchdog>=3.0.0" \
    "mistune>=3.0.0" \
    "chromadb>=0.4.0" \
    "toml>=0.10.0" \
    "pydantic>=2.0.0" \
    "python-multipart>=0.0.6" \
    "aiofiles>=23.0.0" \
    "litellm>=1.35.2" \
    "openai>=1.30.1" \
    "llama-index-core>=0.10.0" \
    "llama-index-llms-litellm" \
    "llama-index-llms-openai" \
    "llama-index-vector-stores-chroma" \
    "llama-index-readers-file>=0.5.0" \
    "llama-index-readers-obsidian>=0.6.0" \
    "llama-index-readers-joplin>=0.5.0" \
    "fastapi-mcp>=0.4.0" \
    "pymerkle>=2.0.1"

echo "Installing CPU-only versions of suspect packages..."
# For sentence-transformers, we need to ensure a CPU-only PyTorch is installed.
# We'll install PyTorch CPU first, then sentence-transformers.
pip install torch==2.3.0+cpu torchvision==0.18.0+cpu torchaudio==2.3.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install "sentence-transformers>=5.1.0"

# mlx-lm does not require the [cuda] extra for CPU-only usage.
pip install "mlx-lm>=0.12.0"

echo "Installing development dependencies..."
pip install \
    "pytest>=7.0.0" \
    "pytest-asyncio>=0.21.0" \
    "pytest-cov>=4.1.0" \
    "pytest-timeout>=2.1.0" \
    "black>=23.0.0" \
    "ruff>=0.1.0" \
    "mypy>=1.5.0" \
    "bandit>=1.7.0" \
    "pre-commit>=3.0.0" \
    "pyfakefs>=5.0.0" \
    "pyfakefs>=5.9.2" \
    "pytest>=8.4.1" \
    "pytest-cov>=6.2.1" \
    "types-toml>=0.10.8.20240310"

echo "All dependencies installed successfully (CPU-only where applicable)."
