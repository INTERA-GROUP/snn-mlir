# SNN Dialect for MLIR

[![CI](https://github.com/INTERA-GROUP/snn-mlir/actions/workflows/ci.yml/badge.svg)](https://github.com/INTERA-GROUP/snn-mlir/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/snn-mlir/badge/?version=latest)](https://snn-mlir.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/snn-mlir)](https://pypi.org/project/snn-mlir/)
[![arXiv](https://img.shields.io/badge/arXiv-2606.09213-b31b1b.svg)](https://arxiv.org/abs/2606.09213)
[![Collaboration Network](https://img.shields.io/badge/Collaboration_Network-Open_Neuromorphic-blue)](https://open-neuromorphic.org/)

An out-of-tree [MLIR](https://mlir.llvm.org/) dialect for Spiking Neural Networks (SNNs), compatible with the [NIR (Neuromorphic Intermediate Representation)](https://neuroir.org/) standard.

The dialect provides type-polymorphic operations that work with both `f32` (float) and quantized (`i8`/`i32`) types, enabling a single IR to target both simulation and hardware-optimized deployments. A reference CPU lowering (`SNNToLinalg`) converts SNN ops to standard `linalg`/`arith` operations that any MLIR-based backend can consume.

A companion Python package (`snn-mlir`, available on [PyPI](https://pypi.org/project/snn-mlir/)) reads any NIR file and emits SNN dialect MLIR text, together with the C sources of a reference CPU runtime. Its `snn-mlir` command takes a trained network from `.nir` to a running binary without writing a line of Python.

---

## Quick start

```bash
git clone <this-repo> snn-mlir
cd snn-mlir
uv sync                                              # install Python env (requires uv)
uv run snn-mlir codegen examples/snn_oxford -q       # NIR → MLIR + C sources
```

This produces `examples/snn_oxford/build/`:

```
network.mlir   ← SNN dialect IR (feed to snn-opt; weights baked in as constant globals)
snn_data.h     ← layer-size constants
input.h        ← input.csv baked into int8_t L0_input[N_STEPS][INPUT_SIZE]
main.c         ← memref descriptor structs + timestep loop
```

Once `snn-opt` and LLVM are built (see [Building the dialect](#building-the-dialect)), one command does the whole thing — codegen, lower, compile, execute:

```bash
uv run snn-mlir run examples/snn_oxford -q
# → examples/snn_oxford/build/results.csv   (one row per timestep)
```

The four verbs:

```
snn-mlir check   <model.nir> [--json]             # is this model supported? (no conversion)
snn-mlir export  <model.nir> [-o OUT.mlir] [-q]   # NIR → SNN dialect MLIR
snn-mlir codegen <folder> [-q]                    # folder → build/ (MLIR + C sources)
snn-mlir run     <folder> [-q]                    # + compile, execute → results.csv
```

`check` is the one to reach for first with an unfamiliar model. It reports every node against
the front-end's rules — and the graph's topology separately — without converting anything, so
you learn what is unsupported up front instead of on the first `export` that stops at it:

```
$ snn-mlir check brailernn.nir
brailernn.nir — 7 nodes

  ok    input       Input    terminal
  ok    fc1         Linear   synapse
  ...

  ERROR   Node 'lif1.lif' has 2 successors — only linear-chain graphs are supported.

not supported: 1 error.
```

It exits non-zero when the model cannot be converted, so it doubles as a CI gate.

A model folder is exactly one `*.nir` plus an `input.csv` (one row per timestep, one column per input channel — the row count is what sets `N_STEPS`). `-q`/`--quantize` selects int8 weights and Q12 fixed-point state; the default is `f32`. Full walk-through: [Quick start](https://snn-mlir.readthedocs.io/en/latest/getting-started/quickstart/).

---

## How it works

```
┌─────────────┐   snn_mlir.export()    ┌─────────────────┐
│  .nir file  │ ─────────────────────► │  network.mlir   │  SNN dialect IR
└─────────────┘                        └─────────────────┘
       │                                        │
       │  codegen_folder()                      │  snn-opt + mlir-opt + mlir-translate
       │  (+ input.csv)                         ▼
       ▼                               ┌─────────────────┐
   ┌────────────────────┐              │  network.ll     │  LLVM IR
   │  snn_data.h        │              └─────────────────┘
   │  input.h           │                       │
   │  main.c            │                       │  llc
   └────────────────────┘                       ▼
              │                        ┌─────────────────┐
              └──────────────────────► │   executable    │ ──► results.csv
                        cc             └─────────────────┘
```

`snn_mlir.export()` converts the NIR graph to SNN dialect MLIR text.
`snn_mlir.codegen_folder()` generates the C runtime files: memref descriptor typedefs, neuron-state buffers, the input array baked from `input.csv`, and a `main.c` timestep loop.
`pipelines/lower_cpu_linux.sh` chains `snn-opt → mlir-opt → mlir-translate` to produce LLVM IR.
`llc` turns that into an object file and a standard C compiler links everything into a self-contained binary.
`snn_mlir.run_folder()` — `snn-mlir run` — drives all of it and captures the output.

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

Requires Python ≥ 3.10. `pip install snn-mlir` gives you the `snn-mlir` command with `export` and
`codegen`; `run` additionally requires the `snn-opt` toolchain (see the build instructions below).

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

### Generating and running the C reference

The same folder-level functions the CLI drives are part of the package:

```python
import snn_mlir

# codegen — model folder → build/ (network.mlir, snn_data.h, input.h, main.c)
build = snn_mlir.codegen_folder(
    "examples/snn_oxford",
    quantize=True,
    index_bits=64,      # 32 for embedded targets; not exposed on the CLI
)

# run — codegen, lower, compile, execute; returns the path to results.csv
results = snn_mlir.run_folder("examples/snn_oxford", quantize=True)
print(results.read_text())
```

The folder must hold exactly one `*.nir` and an `input.csv`; `n_steps` is the CSV's row count, and
its column count must match the network input size. `run_folder` raises a `FileNotFoundError`
listing what is missing if the toolchain is incomplete — `snn_mlir.toolchain_available()` checks
the same thing without raising.

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

Each SNN op covers a family of NIR nodes. The non-leaky nodes (`nir.IF`, `nir.I`) reuse the same ops as their leaky counterparts with the decay set to 1.0 (quantized: `decay_int = 1 << d_scale`), which disables the exponential leak.

| NIR node | SNN op | Notes |
|---|---|---|
| `nir.Linear` | `snn.linear` | No bias |
| `nir.Affine` | `snn.linear` | Bias added as second operand |
| `nir.CubaLIF` | `snn.cubalif` | `cur_decay`, `vol_decay` < 1, derived from `tau_syn`/`tau_mem`/`r` |
| `nir.CubaLI` | `snn.cubali` | `cur_decay`, `vol_decay` < 1, derived from `tau_syn`/`tau_mem`/`r` |
| `nir.LIF` | `snn.lif` | `decay` < 1, derived from `tau`/`r` |
| `nir.IF` | `snn.lif` | `decay = 1.0` by definition; requires `r = 1` |
| `nir.LI` | `snn.li` | `decay` < 1, derived from `tau`/`r` |
| `nir.I` | `snn.li` | `decay = 1.0` by definition; requires `r = 1` |
| _(internal)_ | `snn.rescale` | Inserted between `snn.linear` and neuron ops during quantized export; no NIR equivalent |

Eight NIR node types map today; the table above is the complete list. NIR has no cumulative-current integrate-and-fire node — there is no `nir.CubaIF` or `nir.CubaI` — so `snn.cubalif`/`snn.cubali` are reachable only from their leaky counterparts.

Note that `nir.IF`/`nir.I` are **not** the leaky parsers with the decay forced to 1: they carry no `tau` and no `v_leak`, so `decay = 1` is the definition of the node rather than a derived value, and `r` stops cancelling out of the input gain — which is why they require `r = 1` while `LIF`/`LI` accept any `r`. See [NIR mapping](https://snn-mlir.readthedocs.io/en/latest/python/nir-mapping/) for the derivation.

---

## Examples

Each example is a model folder — one `network.nir` plus an `input.csv` — so both follow the same pattern: `codegen` for the sources, `run` for a finished execution.

### `examples/snn_oxford/`

A two-layer CubaLIF network trained on the Oxford dataset using LAVA-DL, driven by 100 timesteps of 200-channel input:

```
Linear(200→256) → CubaLIF(256) → Linear(256→200) → CubaLIF(200)
```

```bash
uv run snn-mlir run examples/snn_oxford              # float32
uv run snn-mlir run examples/snn_oxford --quantize   # int8 weights, Q12 state
uv run snn-mlir codegen examples/snn_oxford          # sources only, no toolchain needed
```

### `examples/snntorch/`

A decoy network exported from SNNTorch that mixes `Linear`/`Affine` and `LIF`/`CubaLIF` to exercise the pipeline; 25 timesteps of 784-channel random input:

```bash
uv run snn-mlir run examples/snntorch
uv run snn-mlir run examples/snntorch --quantize
```

### Generated files explained

After running either example you will find a `build/` directory with:

| File | Description |
|------|-------------|
| `network.mlir` | SNN dialect IR — the MLIR representation of the network, with weights baked in as `memref.global` constants. Feed this to `snn-opt` and the lowering pipeline. |
| `snn_data.h` | C header: `#define` constants for layer sizes. Include in `main.c`. |
| `main.c` | C harness: MLIR memref descriptor typedefs, neuron state arrays, a timestep loop that calls `_mlir_ciface_snn_forward_step`, and CSV output. |
| `input.h` | `input.csv` baked into `int8_t L0_input[N_STEPS][INPUT_SIZE]`. |

`snn-mlir run` adds `network.ll`, `network.o`, the executable, and `results.csv` — one row per timestep, with no reference comparison. Nothing is cleaned up.

`main.c` is independent of the MLIR toolchain — it is standard C and can be compiled with any C11 compiler once `network.ll` (or a `.o` from it) is available.

---

## Full pipeline (CPU, x86-64)

`snn-mlir run` does all of this for you; here it is by hand, for when you want to inspect or modify a stage:

```bash
# 1. Generate the sources, and point at your LLVM build
uv run snn-mlir codegen examples/snn_oxford
export MLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir

# 2. Lower network.mlir → network.ll (LLVM IR)
bash pipelines/lower_cpu_linux.sh examples/snn_oxford/build/network.mlir

# 3. Compile the IR with the SAME LLVM that built snn-opt, then link with any C compiler
$MLIR_DIR/../../../bin/llc --relocation-model=pic -filetype=obj \
    examples/snn_oxford/build/network.ll -o examples/snn_oxford/build/network.o

cc examples/snn_oxford/build/main.c examples/snn_oxford/build/network.o \
   -o examples/snn_oxford/build/sim -lm

# 4. Run — outputs CSV rows (one per timestep)
./examples/snn_oxford/build/sim
```

The pipeline script chains `snn-opt --convert-snn-to-linalg | mlir-opt <passes> | mlir-translate --mlir-to-llvmir`. See `pipelines/lower_cpu_linux.sh` for the full pass sequence.

> **Don't hand `network.ll` to your system clang.** The IR carries the LLVM version that built `snn-opt`, and an older system clang will reject it. Always go `.ll` → `.o` with the matching `llc`, then let the system compiler see only plain C and an object file.

---

## Repository structure

```
include/SNN/                   Dialect headers and TableGen definitions
  SNNDialect.td / .h           Dialect declaration
  SNNOps.td / .h               Op definitions (ODS format)
  SNNInterfaces.td / .h        Op interfaces (SynapseOpInterface, NeuronOpInterface)
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
  _codegen.py                  C runtime generator: codegen_folder()
  _run.py                      Compile + execute: run_folder(), toolchain detection
  _cli.py                      The snn-mlir command (export / codegen / run)
  nodes/                       One module per NIR node type; NODE_PARSERS registry

python/tests/                  Python unit tests (pytest)

examples/
  snn_oxford/                  LAVA-DL CubaLIF example (network.nir + input.csv)
  snntorch/                    SNNTorch example (network.nir + input.csv)

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
#include "SNN/SNNInterfaces.h"           // SynapseOpInterface / NeuronOpInterface
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

### Matching ops uniformly with the op interfaces

To lower "any synapse layer" or "any spiking neuron" without a `switch` over the concrete op
types, use the two op interfaces the dialect declares natively:

- **`SynapseOpInterface`** on `snn.linear` — activation/weights/accumulator/bias operands plus the
  `getK()` / `getN()` shape of `accumulator = weights @ input`.
- **`NeuronOpInterface`** on `snn.cubalif` / `snn.lif` / `snn.li` / `snn.cubali` — operands,
  fixed-point parameters, and two capability predicates (`hasCurrentStage()` / `producesSpike()`)
  that collapse the four neuron kinds into one uniform reader.

Include `SNN/SNNInterfaces.h`, link `MLIRSNN`, and match with
`OpInterfaceRewritePattern<snn::NeuronOpInterface>` (or `SynapseOpInterface`) instead of the
concrete op. See [Implementing a new lowering pass](docs/dialect/lowering-pass.md) for a worked
example and the full method reference.

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
