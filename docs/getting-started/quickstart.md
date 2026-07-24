# Quick start

Everything below assumes you finished [Installation](installation.md). `export` and `codegen`
need the Python environment alone; `run` additionally needs the `snn-opt` toolchain.

## 1. Run a shipped example

```bash
uv run snn-mlir run examples/snn_oxford           # float32
uv run snn-mlir run examples/snn_oxford -q        # int8 weights + Q12 state
```

This compiles and executes the network, then writes
`examples/snn_oxford/build/results.csv` — one row per timestep. Nothing is compared against a
reference: what the network produced is what you get, to diff however you like.

No toolchain yet? Generate the sources and stop there:

```bash
uv run snn-mlir codegen examples/snn_oxford -q
```

## 2. What you get

Both `codegen` and `run` write a `build/` folder next to the model:

| File | Role |
|---|---|
| `network.mlir` | SNN dialect IR; weights baked in as constant globals |
| `snn_data.h` | Layer-size macros (`N_STEPS`, `INPUT_SIZE`, per-layer sizes) |
| `input.h` | Your `input.csv`, baked into `int8_t L0_input[N_STEPS][INPUT_SIZE]` |
| `main.c` | Memref descriptors, neuron state, timestep loop, CSV output |

`run` adds `network.ll`, `network.o`, the executable, and `results.csv`. Nothing is cleaned up —
every intermediate stays on disk for inspection.

## 3. Bring your own model

A model folder is exactly two files:

```
my_model/
  something.nir     ← exactly one .nir file (any name)
  input.csv         ← one row per timestep, one column per input channel
```

Rules worth knowing before you hit an error message:

- **Exactly one** `.nir` in the folder — two is an error, zero is an error.
- `input.csv` has **no header**. Its **row count sets `n_steps`**,

Then it is the same command as the example:

```bash
uv run snn-mlir run my_model -q
```

Just want the MLIR for a single `.nir`, with no folder and no C?

```bash
uv run snn-mlir export my_model/something.nir             # → my_model/something.mlir
uv run snn-mlir export path/to/net.nir -o build/net.mlir -q
```

## 4. The full CLI surface

```
snn-mlir --version
snn-mlir export  <model.nir> [-o OUT.mlir] [-q]
snn-mlir codegen <folder> [-q]
snn-mlir run     <folder> [-q] [--platform linux]
```

| Flag | Applies to | Default | Effect |
|---|---|---|---|
| `-q`, `--quantize` | all three | off | int8 weights + Q12 fixed-point neuron state, inserting `snn.rescale` where needed. Off means `f32`. |
| `-o`, `--output` | `export` | `<model>.mlir` beside the input | Destination `.mlir` path |
| `--platform` | `run` | `linux` | Reserved for future targets; `linux` is the only value today |

### Quantized or float?

Float (`f32`) is the reference: it reproduces the simulator's numerics most closely and needs
no calibration. Quantized (`-q`) is what embedded targets want — int8 weights and Q12 state,
one power-of-two scale per layer. Start float to confirm the model compiles and behaves, then
switch to `-q` and check the difference in `results.csv` is within your tolerance. See
[Quantization](../python/quantization.md) for the scheme itself.

## 5. The same things from Python

Every verb has a function behind it, for when you would rather script it than shell out.

```python
import snn_mlir

# export — NIR to SNN dialect MLIR
snn_mlir.export("my_model/something.nir", "build/network.mlir", quantize=True)
mlir_text = snn_mlir.to_mlir("my_model/something.nir")        # as a string instead

# codegen — a model folder to build/ (network.mlir + snn_data.h + input.h + main.c)
build = snn_mlir.codegen_folder("my_model", quantize=True)

# run — codegen, compile, execute; returns the path to results.csv
results = snn_mlir.run_folder("my_model", quantize=True)
print(results.read_text())
```

`codegen_folder` takes one extra argument the CLI does not expose: `index_bits` (default `64`),
the width of the memref descriptor index fields — set it to `32` when generating for a 32-bit
embedded target. `run_folder` also accepts `platform="linux"`.

If you want to check the toolchain yourself before calling `run_folder` — to skip a test, say —
`snn_mlir.toolchain_available()` returns a bool instead of raising.

For finer-grained control of the frontend (inspecting or quantizing the parsed layers before
emitting MLIR), see the [API reference](../python/api.md).

## Next steps

For a walk-through of a real network — its topology, what the generated files contain, and how
to read the output — see the examples:
[SNN Oxford (LAVA-DL)](../examples/snn-oxford.md) and [SNNTorch](../examples/snntorch.md).
