# Changelog
## v0.6.0 (2026-08-11)

### Feat

- **codegen**: create codegen for executing recurrent and dot named nodes
- **compiler**: support for recurrent nodes in SNNs.

### Fix

- **cubalif**: reject nir models that do not follow discretization convention

## v0.5.0 (2026-08-04)

### Feat

- **api**: add check function to know if a nir model is compatible with the dialect

## v0.4.0 (2026-07-24)

### Feat

- **run**: detect automatically MLIR and SNN-OPT installation
- **examples**: autogenerate input.h now inputs can be provided as .csv files
- **cli**: added command line integration for snn-mlir. export and codegen

## v0.3.1 (2026-07-21)

### Fix

- **ci**: satisfy ruff check and pin ruff to CI's version

## v0.3.0 (2026-07-21)

### Feat

- **dialect**: add SynapseOpInterface and NeuronOpInterface

### Fix

- **nodes**: reject unsupported v_reset, fix LI parsing, clamp w_scale

## v0.2.0 (2026-06-25)

### Feat

- **python**: emit weights as in-IR memref.global constants

## v0.1.3 (2026-06-22)

### Feat

- **python**: add structured NIR-to-MLIR API

## v0.1.2 (2026-06-18)

### Feat

- **pypi**: add PyPI metadata, badges and frontend install docs

## v0.1.1 (2026-06-09)

### Fix

- **uv**: updates snn-mlir version to 0.1.0

## v0.1.0 (2026-06-01)

## v0.1.0b0 (2026-06-01)

### Feat

- **license**: add license header and license file to git repository
- **main**: first commit

### Fix

- **license**: add license to pyproject.toml

### Refactor

- **readme**: change name of snn-dialect to snn-mlir to match github repository
