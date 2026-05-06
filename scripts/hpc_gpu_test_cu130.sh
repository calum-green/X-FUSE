#!/bin/bash
set -e

# Module name may vary by cluster — check available modules with: module avail cuda
echo "Loading CUDA 13.0..."
module load cuda/13.0

echo "Syncing Python dependencies..."
uv sync

echo "Installing latest PyTorch with CUDA 13.0 wheels..."
uv pip install torch --index-url https://download.pytorch.org/whl/cu130

echo "Running GPU tests..."
uv run pytest -m gpu -v
