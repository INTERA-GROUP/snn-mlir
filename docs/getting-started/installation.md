# Installation

There are two levels of setup, depending on how far down the pipeline you want to go:

1. **Python only** — enough for `snn-mlir export` and `snn-mlir codegen`: read NIR, emit
   `network.mlir` and the C sources. Fast, no compiler build.
2. **Full toolchain** — adds LLVM/MLIR and the `snn-opt` tool, which is what `snn-mlir run`
   needs to compile and execute the model.

Start with the Python setup; add the toolchain when you need to compile and run.

## 1. Python environment

The package targets Python ≥ 3.10 and uses [uv](https://docs.astral.sh/uv/), which manages the
Python version, the virtualenv, and all dependencies (including NIR) for you.

```bash
git clone https://github.com/INTERA-GROUP/snn-mlir.git
cd snn-mlir
uv sync                        # creates .venv and installs everything, NIR included
uv run pre-commit install      # optional: ruff lint + format git hooks
```

Verify it works:

```bash
uv run snn-mlir --version      # the CLI is installed
uv run pytest                  # Python unit tests should all pass
```

You can already generate MLIR and the C runtime from a NIR file:

```bash
uv run snn-mlir export examples/snn_oxford/network.nir   # → examples/snn_oxford/network.mlir
uv run snn-mlir codegen examples/snn_oxford              # → examples/snn_oxford/build/
```

If `build/network.mlir` and `build/main.c` appear, the frontend is working. To go further and
actually compile and run that MLIR (`snn-mlir run`), set up the toolchain below.

## 2. LLVM/MLIR + the dialect (full toolchain)

Needed only to lower `network.mlir` → LLVM IR → executable, which is what `snn-mlir run` does.

### Prerequisites

- CMake ≥ 3.20 and Ninja (`sudo apt-get install ninja-build`)
- A C++17 compiler (GCC ≥ 9 or Clang ≥ 10)
- LLVM/MLIR ≥ 22.1 built with MLIR enabled

### If you already have an MLIR build

Point the build at it and you're done — just set `MLIR_DIR`:

```bash
export MLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir
```

### If you need to build LLVM/MLIR

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
cmake -G Ninja -S llvm -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_INSTALL_UTILS=ON \
  -DCMAKE_INSTALL_PREFIX=$HOME/mlir-install
cmake --build build --target install
```

### Build the dialect (`snn-opt`)

The helper script does it in one step:

```bash
bash scripts/build_snn_dialect.sh
# Produces build/bin/snn-opt
```

Or manually:

```bash
cmake -G Ninja -B build \
  -DMLIR_DIR=$HOME/mlir-install/lib/cmake/mlir \
  -DLLVM_EXTERNAL_LIT=$HOME/mlir-install/bin/llvm-lit
cmake --build build --target snn-opt
```

### Verify the toolchain

```bash
./build/bin/snn-opt --help              # the tool runs
ninja -C build check-snn                # MLIR lit tests (FileCheck over test/Dialect/SNN/*.mlir)
uv run snn-mlir run examples/snn_oxford # the whole pipeline, end to end
```

If `check-snn` is green and `run` writes `examples/snn_oxford/build/results.csv`, you have a
working end-to-end install.

!!! tip "No environment variables needed for an in-repo build"
    `snn-mlir run` finds `snn-opt` at `build/bin/snn-opt` and reads the LLVM tool paths from the
    `MLIR_DIR` recorded in `build/CMakeCache.txt` — so it always uses the same LLVM that built
    `snn-opt`. Set `SNN_OPT` (the `snn-opt` binary), `MLIR_DIR` (your LLVM build's
    `lib/cmake/mlir`), or `CC` (the C compiler used for the final link) only to override that.

Head to the [Quick start](quickstart.md) to run your own model, or to the
[examples](../examples/snn-oxford.md) for a walk-through of a real network.
