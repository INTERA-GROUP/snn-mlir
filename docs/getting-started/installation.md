# Installation

There are two levels of setup, depending on how far down the pipeline you want to go:

1. **Python only** — enough to read NIR and generate `network.mlir`. Fast, no compiler build.
2. **Full toolchain** — adds LLVM/MLIR and the `snn-opt` tool so you can lower all the way to
   an executable.

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
uv run python -c "import snn_mlir; print(snn_mlir.__name__, 'ok')"
uv run pytest                  # Python unit tests should all pass
```

You can already generate MLIR from a NIR file:

```bash
uv run python examples/snn_oxford/run.py            # writes examples/snn_oxford/build/network.mlir
```

If `build/network.mlir` appears, the frontend is working. To go further and actually compile
that MLIR, set up the toolchain below.

## 2. LLVM/MLIR + the dialect (full toolchain)

Needed only to lower `network.mlir` → LLVM IR → executable.

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
./build/bin/snn-opt --help     # the tool runs
ninja -C build check-snn       # MLIR lit tests (FileCheck over test/Dialect/SNN/*.mlir)
```

If `check-snn` is green, you have a working end-to-end install. Head to the
[examples](../examples/snn-oxford.md) to lower and run a real network.

## Repository layout

```
include/SNN/                   Dialect headers and TableGen definitions
  SNNDialect.td / .h           Dialect declaration
  SNNOps.td / .h               Op definitions (ODS format)
  Conversion/
    SNNToLinalg.h              Public header for the CPU lowering pass

lib/Dialect/SNN/               Dialect implementation (auto-generated + custom)
lib/Conversion/SNNToLinalg/    CPU lowering: snn.* → linalg/arith

tools/snn-opt/                 Standalone opt tool (dialect + CPU lowering)

pipelines/
  lower_cpu_linux.sh           Lower SNN dialect → LLVM IR on x86-64 Linux

test/Dialect/SNN/              Roundtrip and lowering tests (llvm-lit)

python/snn_mlir/               pip-installable Python package
  _api.py                      Public API: to_mlir(), export()
  _graph.py                    NIR graph walker and quantizer
  _emit.py                     MLIR text emitter
  nodes/                       One module per NIR node type; NODE_PARSERS registry

python/tests/                  Python unit tests (pytest)

examples/
  _codegen.py                  C runtime file generator (snn_data.h/c + main.c)
  snn_oxford/                  LAVA-DL CubaLIF example (network.nir + run.py)
  snntorch/                    SNNTorch example (network.nir + run.py)

scripts/
  build_snn_dialect.sh         One-time build of snn-opt
```
