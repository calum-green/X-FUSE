# CI/CD Design — X-FUSE

**Date:** 2026-05-06
**Author:** Calum Green
**Status:** Approved

---

## Overview

Lightweight CI/CD setup for X-FUSE, a solo PhD research project. Automates linting and CPU tests on every push/PR via GitHub Actions free runners. GPU tests are run manually on HPC via shell scripts. No Docker, no Azure VM, no ACR required.

---

## Trigger Summary

| Event | What runs | Where |
|---|---|---|
| Every push / PR | Lint + CPU tests | GitHub free runner |
| Before every commit | black + flake8 | Local (pre-commit) |
| Before merging (manual) | GPU tests | HPC interactive job |

---

## Files

### New Files

| File | Purpose |
|---|---|
| `.github/workflows/lint-cpu.yml` | GitHub Actions workflow |
| `.pre-commit-config.yaml` | Local pre-commit hooks |
| `scripts/hpc_gpu_test_cu124.sh` | Manual GPU test script — CUDA 12.4 |
| `scripts/hpc_gpu_test_cu130.sh` | Manual GPU test script — CUDA 13.0 |

### Modified Files

| File | Change |
|---|---|
| `pyproject.toml` | Add `pre-commit` to dev dependencies; register `gpu` pytest marker |

---

## GitHub Actions Workflow

**File:** `.github/workflows/lint-cpu.yml`

**Triggers:**
- Push to any branch
- Pull request targeting `main`

**Job 1: lint**
- Runner: `ubuntu-latest` (GitHub free)
- Steps: install uv → `uv sync --extra cpu --extra dev` → `black --check . --line-length 88` → `flake8 x_fuse/ tests/`

**Job 2: test** (only runs if lint passes)
- Runner: `ubuntu-latest` (GitHub free)
- Steps: install uv → `uv sync --extra cpu --extra dev` → `pytest -m "not gpu" --cov=x_fuse --cov-report=xml`
- Coverage report generated, no hard failure threshold

---

## Pre-commit

**File:** `.pre-commit-config.yaml`

Runs `black` and `flake8` automatically before every `git commit`. Activated once with `pre-commit install`. No pytest — keeps commit hooks fast so they are never disabled.

---

## pytest Markers

**Registered in `pyproject.toml`:**

```toml
[tool.pytest.ini_options]
markers = ["gpu: marks tests requiring a CUDA-enabled GPU"]
```

GPU-specific tests decorated with `@pytest.mark.gpu`:
- Tests that verify tensors are on the correct device
- Tests that verify data is `torch.Tensor` type on CUDA

CI excludes GPU tests via `-m "not gpu"`. HPC scripts run only GPU tests via `-m gpu`.

---

## HPC GPU Test Scripts

Two scripts differing only in CUDA version and torch wheel.

**`scripts/hpc_gpu_test_cu124.sh`** — for CUDA 12.4 nodes:
1. `module load cuda/12.4`
2. `uv sync`
3. `uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124`
4. `pytest -m gpu`

**`scripts/hpc_gpu_test_cu130.sh`** — for CUDA 13.0 nodes (newer GPUs, longer queue):
1. `module load cuda/13.0`
2. `uv sync`
3. `uv pip install torch --index-url https://download.pytorch.org/whl/cu130` (latest torch, no pin — torch<=2.6 has no CUDA 13.0 wheels)
4. `pytest -m gpu`

**Usage:**
```bash
# SSH into HPC interactive GPU job, then:
bash scripts/hpc_gpu_test_cu124.sh
# or
bash scripts/hpc_gpu_test_cu130.sh
```

---

## Decisions and Rationale

| Decision | Rationale |
|---|---|
| No Docker | HPC uses Singularity not Docker; uv.lock provides Python reproducibility |
| No ACR | No GPU VM consumer for images; unnecessary cost |
| No coverage failure threshold | Research codebase with GPU-dependent code that is hard to test on CPU; threshold may be added later |
| Two separate HPC scripts | CUDA 13.0 requires newer torch than CUDA 12.4; keeping scripts separate avoids conditional logic |
| lint gates test | No point running tests against unformatted code; saves runner minutes |
| pre-commit local only | Running pytest in pre-commit is too slow; CI handles test automation |

---

## Out of Scope

- Azure GPU VM CI (dropped — quota restrictions)
- Docker images and ACR (dropped — unnecessary without GPU VM)
- Coverage failure threshold (deferred — may add later)
- `diff-cover` (deferred — may add later)
