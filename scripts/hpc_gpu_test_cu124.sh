#!/bin/bash
set -e

# Module name may vary by cluster — check available modules with: module avail cuda
echo "Loading CUDA 12.4..."
module load cuda/12.4

echo "Syncing Python dependencies..."
uv sync

echo "Installing PyTorch with CUDA 12.4 wheels..."
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

echo "Running GPU tests..."
uv run pytest -m gpu -v
