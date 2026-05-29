## v0.2.0a0 (2026-05-28)

### Feat

- **commitizen**: add commitizen to repo
- **pre-commit**: Add pre-commits for ensuring ruff formating
- **uv**: add uv environment as package manager
- **test**: add pytests to the python export module
- **snn-mlir**: add python module for exporting NIR into MLIR code
- **test**: add compiler tests for validation
- **CubaLi**: add CubaLi neurons. Using the output with i32. No spikes
- **LiF**: add LiF and Li mapping of Neurons
- **affine**: add bias to Linear layers. Not called affine for MLIR compatibility
- **snn-dialect**: first NIR modules mapped to lowerization (Linear, CubaLIF, Rescale)

### Fix

- **warnings**: remove warnings for b.create deprecated in MLIR
- **Affine**: fix bias of 8bit to 32bit according to standard Tensorflow

### Refactor

- **ruff**: add last empty line to files
- **ruff**: make ruff happy
