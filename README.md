# SNN Dialect for MLIR

[![CI](https://github.com/INTERA-GROUP/snn-mlir/actions/workflows/ci.yml/badge.svg)](https://github.com/INTERA-GROUP/snn-mlir/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/snn-mlir/badge/?version=latest)](https://snn-mlir.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/snn-mlir)](https://pypi.org/project/snn-mlir/)
[![arXiv](https://img.shields.io/badge/arXiv-2606.09213-b31b1b.svg)](https://arxiv.org/abs/2606.09213)
[![Collaboration Network](https://img.shields.io/badge/Collaboration_Network-Open_Neuromorphic-blue)](https://open-neuromorphic.org/)

An out-of-tree [MLIR](https://mlir.llvm.org/) dialect for Spiking Neural Networks (SNNs), compatible with the [NIR (Neuromorphic Intermediate Representation)](https://neuroir.org/) standard.

The dialect provides type-polymorphic operations that work with both `f32` (float) and quantized (`i8`/`i32`) types, enabling a single IR to target both simulation and hardware-optimized deployments. A reference CPU lowering (`SNNToLinalg`) converts SNN ops to standard `linalg`/`arith` operations that any MLIR-based backend can consume.

A companion Python package (`snn-mlir`, available on [PyPI](https://pypi.org/project/snn-mlir/)) reads any NIR file and emits SNN dialect MLIR text, ready to feed into the `snn-opt` lowering toolchain. (The C runtime files used in the examples are generated separately by `examples/_codegen.py`, which is not part of the installable package.)

---

## Quick start

```bash
git clone <this-repo> snn-mlir
cd snn-mlir
uv sync                          # install Python env (requires uv)
uv run python examples/snn_oxford/run.py --quantize
```

This produces `examples/snn_oxford/build/`:

```
network.mlir   ← SNN dialect IR (feed to snn-opt; weights baked in as constant globals)
snn_data.h     ← layer-size constants
main.c         ← memref descriptor structs + timestep loop
input.h        ← pre-baked input data (copied from examples/snn_oxford/)
```

To lower all the way to an executable (requires `snn-opt` and LLVM built — see [Building the dialect](#building-the-dialect)):

```bash
export MLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir
bash pipelines/lower_cpu_linux.sh examples/snn_oxford/build/network.mlir
# → examples/snn_oxford/build/network.ll

clang examples/snn_oxford/build/network.ll \
      examples/snn_oxford/build/main.c \
      -o examples/snn_oxford/build/sim

./examples/snn_oxford/build/sim
```

---

## How it works

```
┌─────────────┐   snn_mlir.export()    ┌─────────────────┐
│  .nir file  │ ─────────────────────► │  network.mlir   │  SNN dialect IR
└─────────────┘                        └─────────────────┘
                                                │
       _codegen.export()                        │  snn-opt + mlir-opt
              │                                 ▼
              ▼                        ┌─────────────────┐
   ┌────────────────────┐              │  network.ll     │  LLVM IR
   │  snn_data.h / .c   │              └─────────────────┘
   │  main.c            │                       │
   │  input.h (copied)  │                       │  clang
   └────────────────────┘                       ▼
              │                        ┌─────────────────┐
              └──────────────────────► │   executable    │
                        link           └─────────────────┘
```

`snn_mlir.export()` converts the NIR graph to SNN dialect MLIR text.
`_codegen.export()` (in `examples/_codegen.py`) generates the C runtime files: weight arrays, memref descriptor typedefs, and a `main.c` timestep loop.
`pipelines/lower_cpu_linux.sh` chains `snn-opt → mlir-opt → mlir-translate` to produce LLVM IR.
A standard C compiler links everything into a self-contained binary.

---

## Python package (`snn-mlir`)

### Installation

```bash
# With uv (recommended — handles Python version and virtualenv)
uv sync

# Or with pip, from source
pip install .

# Or, as a back-up, the Python frontend only from PyPI
pip install snn-mlir
```

Requires Python ≥ 3.10. Note that `pip install snn-mlir` provides only the
NIR-to-MLIR frontend; lowering the emitted MLIR additionally requires the `snn-opt`
toolchain (see the build instructions below).

### API

```python
import snn_mlir

# Convert a NIR file to SNN dialect MLIR text
mlir_text = snn_mlir.to_mlir("network.nir")                 # float32
mlir_text = snn_mlir.to_mlir("network.nir", quantize=True)  # int8 + Q12

# Write directly to a file
snn_mlir.export("network.nir", "build/network.mlir", quantize=True)
```

`to_mlir` returns a string containing the complete MLIR module, ready to pipe into `snn-opt`.

For finer control, the same pipeline is exposed one stage at a time. This lets you inspect or
quantize the parsed `NodeInfo` layers — or feed them to your own code generation — before emitting
MLIR (`to_mlir` is exactly these three composed):

```python
layers = snn_mlir.parse_graph("network.nir")   # ordered list[NodeInfo]
snn_mlir.quantize_layers(layers)               # in-place; call at most once
mlir_text = snn_mlir.mlir_from_layers(layers, quantize=True)
```

See [`docs/python/api.md`](docs/python/api.md) for the full reference.

### Generating C runtime files

The `examples/_codegen.py` module (not part of the pip-installable package) generates the C side:

```python
import sys
sys.path.insert(0, "examples/")
import _codegen

_codegen.export(
    "network.nir",
    "build/",                # output directory
    quantize=True,
    n_steps=100,
    index_bits=64,           # 32 for embedded targets
    input_file="input.h",    # pre-baked input data
)
# Writes: build/snn_data.h, build/main.c
```

### Extending: `NODE_PARSERS`

`NODE_PARSERS` is the single registry mapping NIR node types to handler functions. All other per-node behavior — quantization, MLIR emission, classification traits — lives on the `NodeInfo` subclass itself, so adding a new NIR node type requires three steps:

**1. Create a `NodeInfo` subclass:**

```python
from snn_mlir.nodes import NodeInfo
from dataclasses import dataclass

@dataclass
class MyNodeInfo(NodeInfo):
    name: str
    size: int

    # Classification traits are read-only properties on NodeInfo; override
    # the ones that apply (they default to False on the base class).
    @property
    def is_neuron(self) -> bool:
        return True

    # Override quantize() if the node has quantizable parameters (no-op by
    # default). Called once per layer before MLIR emission in quantized mode.
    def quantize(self) -> None:
        ...

    def emit_mlir(self, input_var, is_last, quantize):
        # Return (list_of_mlir_lines, output_var_name)
        ...
```

**2. Write a parser function:**

```python
import nir
def parse_mynode(node: nir.MyNode, name: str) -> MyNodeInfo:
    return MyNodeInfo(name=name, size=node.output_shape[0])
```

**3. Register it:**

```python
from snn_mlir.nodes import NODE_PARSERS
NODE_PARSERS[nir.MyNode] = parse_mynode
```

---

## Ops

| Op | States | Output | Summary |
|---|---|---|---|
| `snn.linear` | — | `f32`/`i32` | Matrix-vector synapse layer (`weights @ input → output`) |
| `snn.rescale` | — | `i32` | Per-edge requantization shift to align quantization scales |
| `snn.cubalif` | current, voltage | `f32`/`i8` | Current-based leaky integrate-and-fire: two-state dynamics with threshold and voltage reset |
| `snn.cubali` | current, voltage | `f32`/`i32` | Current-based leaky integrator: two-state dynamics, continuous voltage output (no threshold) |
| `snn.lif` | voltage | `f32`/`i8` | Leaky integrate-and-fire: single-state dynamics with threshold and voltage reset |
| `snn.li` | voltage | `f32`/`i32` | Leaky integrator: single-state dynamics, continuous voltage output (no threshold) |

All ops are `memref`-based and carry explicit type information, making them directly inspectable and transformable by standard MLIR passes.

Spike-output ops (`snn.cubalif`, `snn.lif`) emit binary activations (`f32` 0/1 or `i8` 0/1).
Voltage-output ops (`snn.cubali`, `snn.li`) emit continuous membrane potential and are used as the final layer in regression or readout networks.

`snn.rescale` is inserted automatically between `snn.linear` and neuron ops in quantized mode to align the two quantization scales. It has no NIR equivalent — the Python export layer inserts it.

---

## NIR node mapping

Each SNN op covers a family of NIR nodes. Integrate-and-fire variants (`nir.CubaIF`, `nir.IF`) map to the same op as their leaky counterparts with decay set to 1.0 (quantized: `decay_int = 1 << d_scale`), which disables the exponential leak.

| NIR node | SNN op | Notes |
|---|---|---|
| `nir.Linear` | `snn.linear` | No bias |
| `nir.Affine` | `snn.linear` | Bias added as second operand |
| `nir.CubaLIF` | `snn.cubalif` | `cur_decay`, `vol_decay` < 1 |
| `nir.CubaIF` | `snn.cubalif` | `cur_decay = vol_decay = 1.0` (no leak) |
| `nir.CubaLI` | `snn.cubali` | `cur_decay`, `vol_decay` < 1 |
| `nir.CubaI` | `snn.cubali` | `cur_decay = vol_decay = 1.0` (no leak) |
| `nir.LIF` | `snn.lif` | `decay` < 1 |
| `nir.IF` | `snn.lif` | `decay = 1.0` (no leak) |
| `nir.LI` | `snn.li` | `decay` < 1 |
| `nir.I` | `snn.li` | `decay = 1.0` (no leak) |
| _(internal)_ | `snn.rescale` | Inserted between `snn.linear` and neuron ops during quantized export; no NIR equivalent |

---

## Examples

Both examples follow the same pattern: run `run.py` to generate the build artefacts, then compile and run.

### `examples/snn_oxford/`

A two-layer CubaLIF network trained on the Oxford dataset using LAVA-DL:

```
Linear(200→256) → CubaLIF(256) → Linear(256→200) → CubaLIF(200)
```

```bash
# Generate MLIR + C files
uv run python examples/snn_oxford/run.py              # float32
uv run python examples/snn_oxford/run.py --quantize   # int8 weights, Q12 state
uv run python examples/snn_oxford/run.py --n-steps 50 # fewer timesteps
```

### `examples/snntorch/`

A network exported from SNNTorch:

```bash
uv run python examples/snntorch/run.py
uv run python examples/snntorch/run.py --quantize
```

### Generated files explained

After running either example you will find a `build/` directory with:

| File | Description |
|------|-------------|
| `network.mlir` | SNN dialect IR — the MLIR representation of the network, with weights baked in as `memref.global` constants. Feed this to `snn-opt` and the lowering pipeline. |
| `snn_data.h` | C header: `#define` constants for layer sizes. Include in `main.c`. |
| `main.c` | C harness: MLIR memref descriptor typedefs, neuron state arrays, a timestep loop that calls `_mlir_ciface_snn_forward_step`, and CSV output. |
| `input.h` | Pre-baked input data (copied from the example directory). Provides `L0_input[N_STEPS][INPUT_SIZE]`. |

`main.c` is independent of the MLIR toolchain — it is standard C and can be compiled with any C11 compiler once `network.ll` (or a `.o` from it) is available.

---

## Full pipeline (CPU, x86-64)

After generating the build artefacts with `run.py`, lower to LLVM IR and compile:

```bash
# 1. Set MLIR_DIR to your LLVM build
export MLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir

# 2. Lower network.mlir → network.ll (LLVM IR)
bash pipelines/lower_cpu_linux.sh examples/snn_oxford/build/network.mlir

# 3. Compile everything to an executable
clang examples/snn_oxford/build/network.ll \
      examples/snn_oxford/build/main.c \
      -o examples/snn_oxford/build/sim

# 4. Run — outputs CSV rows (one per timestep)
./examples/snn_oxford/build/sim
```

The pipeline script chains `snn-opt --convert-snn-to-linalg | mlir-opt <passes> | mlir-translate --mlir-to-llvmir`. See `pipelines/lower_cpu_linux.sh` for the full pass sequence.

---

## Repository structure

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
  _api.py                      Public API: to_mlir(), export(), parse_graph(), quantize_layers(), mlir_from_layers()
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

---

## Development setup

### Prerequisites

- CMake ≥ 3.20, Ninja (`sudo apt-get install ninja-build`)
- C++17 compiler (GCC ≥ 9 or Clang ≥ 10)
- LLVM/MLIR ≥ 22.1 built with MLIR enabled (see below)
- [uv](https://docs.astral.sh/uv/) for Python 3.10+

### Install the Python environment

```bash
uv sync                        # creates .venv and installs all dev dependencies
uv run pre-commit install      # install git hooks (ruff lint + format on every commit)
```

### Building LLVM/MLIR

If you do not have an MLIR installation, build it from source:

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

### Building the dialect

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

### Running the tests

```bash
# Python unit tests
uv run pytest

# MLIR lit tests (requires snn-opt built — runs FileCheck on all test/Dialect/SNN/*.mlir)
ninja -C build check-snn
```

---

## Using the dialect in your own project

Add this repo as a subdirectory (or git submodule) and consume the CMake targets:

```cmake
add_subdirectory(third_party/snn-mlir)

target_include_directories(MyPass PRIVATE
  ${CMAKE_SOURCE_DIR}/third_party/snn-mlir/include
  ${CMAKE_BINARY_DIR}/third_party/snn-mlir/include
)

target_link_libraries(MyPass
  MLIRSNN          # dialect library
  MLIRSNNToLinalg  # CPU lowering pass (optional)
)
```

In your pass source:

```cpp
#include "SNN/SNNOps.h"
#include "SNN/Conversion/SNNToLinalg.h"  // if using the CPU lowering
```

---

## Implementing a new lowering pass

`lib/Conversion/SNNToLinalg/SNNToLinalg.cpp` is the reference implementation. To target a new backend:

**1. Create the pass files:**

```
include/SNN/Conversion/SNNToMyBackend.h
lib/Conversion/SNNToMyBackend/SNNToMyBackend.cpp
lib/Conversion/SNNToMyBackend/CMakeLists.txt
```

**2. Declare your pass in the header:**

```cpp
#include "mlir/Pass/Pass.h"
#include <memory>

namespace snn {
  std::unique_ptr<mlir::Pass> createConvertSNNToMyBackendPass();
  void registerConvertSNNToMyBackendPass();
} // namespace snn
```

**3. Implement a rewrite pattern per op:**

```cpp
#include "SNN/SNNOps.h"

struct LowerLinear : public OpRewritePattern<snn::LinearOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(snn::LinearOp op,
                                PatternRewriter &rewriter) const override {
    // Replace op with your backend calls
    rewriter.eraseOp(op);
    return success();
  }
};
```

**4. Wire up the pass:**

```cpp
struct ConvertSNNToMyBackendPass
    : public PassWrapper<ConvertSNNToMyBackendPass, OperationPass<ModuleOp>> {

  StringRef getArgument() const override { return "convert-snn-to-mybackend"; }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerLinear, LowerRescale, LowerCubaLIF>(&getContext());

    ConversionTarget target(getContext());
    target.addIllegalDialect<snn::SNNDialect>();
    target.addLegalDialect</* your dialects */>();

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};
```

**5. Register in CMake** using `add_mlir_conversion_library()` — see `lib/Conversion/SNNToLinalg/CMakeLists.txt` as a template.

---

## Limitations

The current implementation covers feedforward, fully-connected SNN topologies. The following are known constraints:

**1-D activations only.** All ops (`snn.linear`, `snn.cubalif`, `snn.lif`, etc.) require 1-D activation vectors — `memref<Nxf32>` or `memref<Nxi32>`. Neuron populations are treated as flat arrays, not spatial maps. The verifiers enforce this explicitly, so feeding a 2-D feature map will produce a clear error rather than silent miscompilation.

**No convolutional ops.** NIR nodes such as `nir.Conv2d`, `nir.AvgPool2d`, and `nir.SumPool2d` operate on `[channels, height, width]` feature maps and have no equivalent SNN op yet. Supporting them requires new ops (e.g. `snn.conv2d`). The neuron dynamics ops are already rank-agnostic at the lowering level — extending them to N-D is straightforward once the convolutional synapse op exists.

**Linear-chain graphs only.** The Python graph walker (`_graph.parse_graph`) follows a single path from `input` to `output`. Branching, residual connections, and recurrent edges are not supported.

**Batch size 1.** There is no batched-inference mode. Each call to the compiled function processes one input sample. Batching would require 2-D activation memrefs, which is blocked by the 1-D constraint above.

**Uniform neuron parameters per layer.** All neurons in a layer share the same decay constants and threshold. Per-neuron parameter arrays are not yet supported.

---

## Contributing

Contributions are welcome. Please follow these guidelines:

- Run `uv run pre-commit install` once after cloning — hooks enforce ruff lint and formatting on every commit
- Run `uv run pytest` before opening a PR — all Python unit tests must pass
- Keep ops type-polymorphic (float and quantized must work through the same op)
- New ops must have an `assemblyFormat` for human-readable `.mlir` output
- Add a roundtrip test in `test/Dialect/SNN/` for any new op
- New NIR node types belong in `python/snn_mlir/nodes/` with a matching entry in `NODE_PARSERS`; put quantization in the class's `quantize()` method
- Follow MLIR naming conventions: `add_mlir_dialect_library`, `add_mlir_conversion_library`, `MLIR` prefix on CMake targets

---

## Citation

A companion paper describing snn-mlir is published on arXiv. If you use snn-mlir in your research, please cite the white paper directly:

```bibtex
@misc{gener2026snnmlirmlirdialectcompiling,
      title={SNN-MLIR: An MLIR Dialect for Compiling Neuromorphic SNNs from NIR to Bare-Metal C}, 
      author={Alejandro García Gener and Alvaro Rollón de Pinedo},
      year={2026},
      eprint={2606.09213},
      archivePrefix={arXiv},
      primaryClass={cs.PL},
      url={https://arxiv.org/abs/2606.09213}, 
}
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
