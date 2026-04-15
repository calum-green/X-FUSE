# Installation Guide

This guide covers installing X-FUSE with support for both macOS (CPU) and Linux (CUDA) systems for reproducible environments.

## Prerequisites

### Required
- **Python 3.10** or higher (3.10, 3.11, 3.12 supported)
- **uv** package manager ([install here](https://docs.astral.sh/uv/getting-started/installation/))

### Optional
- **Git** (for cloning the repository)
- **CUDA Toolkit 12.1** (Linux only, for GPU acceleration)

## Quick Start

### macOS (CPU-only)
```bash
# Clone the repository
git clone <repository-url>
cd X-FUSE

# Install dependencies (CPU-only)
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Linux (CUDA 12.1)
```bash
# Clone the repository
git clone <repository-url>
cd X-FUSE

# Install dependencies with CUDA support
uv sync --extra cuda

# Activate virtual environment
source .venv/bin/activate
```

## Detailed Installation

### 1. Clone and Navigate

```bash
git clone <repository-url>
cd X-FUSE
```

### 2. Install with uv

#### For macOS (CPU-only):
```bash
# Default installation (CPU PyTorch)
uv sync

# This installs:
# - featup (with CPU support)
# - hr_dv2 (with CPU support)
# - All dependencies using macOS-compatible wheels
```

#### For Linux (CUDA 12.1):
```bash
# Installation with CUDA support
uv sync --extra cuda

# This installs:
# - featup (with CUDA 12.1 support and extensions)
# - hr_dv2 (with CUDA 12.1 support)
# - CUDA-enabled PyTorch wheels
```

#### For Linux (CPU fallback):
```bash
# If CUDA is not available or preferred
uv sync --extra cpu
```

### 3. Verify Installation

```bash
# Activate the environment
source .venv/bin/activate

# Test imports
python -c "import torch; import featup; import hr_dv2; print('✓ All packages installed successfully')"

# Check PyTorch CUDA availability (Linux)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Lock Files and Reproducibility

### Generate a Lock File

Lock files ensure reproducible installations across different machines and time:

```bash
# Generate lock file for current platform
uv lock

# This creates `uv.lock` - commit this to version control
git add uv.lock
git commit -m "Add reproducible lock file"
```

### Use Existing Lock File

When cloning on another machine:

```bash
# uv automatically uses uv.lock if it exists
uv sync

# Force use of lock file
uv sync --locked
```

## Development Setup

### Install with Development Dependencies

```bash
# macOS development setup
uv sync --extra dev

# Linux development setup with CUDA
uv sync --extra dev --extra cuda
```

### Development Tools Included
- **pytest** - Unit testing
- **pytest-cov** - Code coverage
- **pytest-benchmark** - Performance benchmarking
- **ipython** - Enhanced Python REPL (FeatUp only)
- **jupyter** - Jupyter notebooks (FeatUp only)

### Run Tests

```bash
# Activate environment
source .venv/bin/activate

# Run all tests
pytest

# Run with coverage
pytest --cov

# Run benchmarks
pytest --benchmark-only
```

## Troubleshooting

### Issue: "ModuleNotFoundError" after installation

**Solution:**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall packages
uv sync --force
```

### Issue: CUDA-related errors on Linux

**Solutions:**
1. Check CUDA toolkit installation:
   ```bash
   nvidia-smi
   nvcc --version
   ```

2. Ensure CUDA 12.1 compatibility:
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

3. Fallback to CPU:
   ```bash
   uv sync --extra cpu
   ```

### Issue: macOS M1/M2 ARM64 specific issues

**Solution:** uv automatically selects the correct ARM64 wheels. If issues persist:

```bash
# Force reinstall native wheels
rm -rf .venv
uv sync --force
```

### Issue: Different results between Mac and Linux

**Ensure reproducibility:**
1. Use the same `uv.lock` file on both systems
2. Check Python version: `python --version`
3. Verify torch CUDA availability matches expected platform

## Platform Comparison

| Feature | macOS | Linux |
|---------|-------|-------|
| PyTorch | CPU-only | CUDA 12.1 |
| CUDA Extensions (FeatUp) | ❌ Not built | ✅ Built |
| Installation | `uv sync` | `uv sync --extra cuda` |
| Performance | CPU-bound | GPU-accelerated |
| Reproducible | ✅ Yes | ✅ Yes |

## Advanced Usage

### Override PyTorch Version

```bash
# Use different CUDA version (e.g., CUDA 12.4)
uv sync --override torch=torch[cu124]

# Use development version
uv sync --override torch=torch[dev,cu121]
```

### Clean Installation

```bash
# Remove virtual environment and lock file
rm -rf .venv uv.lock

# Fresh install
uv sync
```

### Check Dependency Tree

```bash
# Show all installed packages
pip list

# Show dependency tree (if pipdeptree installed)
pip install pipdeptree
pipdeptree
```

## Continuous Integration / Automation

### GitHub Actions Example

```yaml
- name: Install X-FUSE on macOS
  if: runner.os == 'macos'
  run: |
    uv sync --locked

- name: Install X-FUSE on Linux with CUDA
  if: runner.os == 'linux'
  run: |
    uv sync --locked --extra cuda
```

## Support

For issues with:
- **Installation**: Check the [uv documentation](https://docs.astral.sh/uv/)
- **FeatUp package**: See [FeatUp GitHub](https://github.com/mhamilton723/FeatUp)
- **HR-Dv2 package**: See [HR-Dv2 GitHub](https://github.com/tldr-group/HR-Dv2)

## Next Steps

After successful installation:
1. Check out example notebooks in `FeatUp/example_usage.ipynb`
2. Run tests: `pytest`
3. Review package documentation in respective READMEs
4. Explore the Gradio app: `python FeatUp/gradio_app.py` (FeatUp)
