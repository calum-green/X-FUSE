# Loaders Design

**Date:** 2026-05-12  
**Status:** Approved

## Overview

`x_fuse/loaders.py` provides all data loading for the X-FUSE pipeline. It is structured in two layers: generic I/O functions and dataset-specific wrappers. Notebooks and scripts import only the dataset-specific wrappers.

## Architecture

### Layer 1 — Generic loaders

Two functions handle all file I/O. They accept all parameters explicitly and are the only place that touches `h5py` and `numpy` directly.

| Function | Signature | Returns |
|---|---|---|
| `load_xct` | `(path, recon, crop)` | `np.ndarray` |
| `load_xrdct_phase` | `(phase_folder, phases, shape, crop)` | `list[np.ndarray]` |

- `load_xct` reads an `.h5` file and returns the XCT volume transposed to `(Y, X, Z)` order.
- `load_xrdct_phase` iterates over phase folders, reads `.nxs` files, and returns a list of phase arrays with shape `(n_files, H, W)`.

### Layer 2 — Dataset-specific wrappers

One function (or pair of functions) per dataset. Each wrapper calls the generic layer with hardcoded dataset parameters: crop slices, volume shape, reconstruction method, and any pre/post-processing (e.g. `np.flip`).

Current wrappers:
- `load_diad_xct_zn13x(xct_path)` — DIAD XCT, Astra reconstruction, crop `[150:2000, 350:2200, 350:2200]`
- `load_diad_xrdct(phase_folder, phases)` — DIAD XRDCT, shape `(21, 20, 20)`, crop `slice(5, -1)`, flip axis 2
- `load_i13_xct(...)` — I13 XCT (parameters TBD by dataset)

### Adding a new dataset

Add one or two wrapper functions at the bottom of `loaders.py`. The generic layer does not need to change unless the file format differs fundamentally.

## Usage

```python
from x_fuse.loaders import load_diad_xct_zn13x, load_diad_xrdct

xct = load_diad_xct_zn13x("/path/to/data.h5")
phases = load_diad_xrdct("/path/to/phases", phases=["ZnO", "Zn13X"])
```

## Implementation approach

The existing `loaders.py` already has the right structure. Implementation is a polish pass on what is written — fixing bugs, tidying signatures, and ensuring consistent style. No functions should be rewritten from scratch.

## Design decisions

- **Flat file, not subpackage** — few datasets expected, so splitting into `loaders/diad.py` etc. adds indirection without benefit.
- **Explicit imports** — callers use `from x_fuse.loaders import <function>` rather than re-exports from `__init__.py`, keeping the source of each function clear.
- **Wrappers, not config dicts** — hardcoding parameters in functions is more readable than a config-dict lookup at this scale.
