# SNN Oxford (LAVA-DL)

A complete, end-to-end example: a two-layer CubaLIF network trained with **LAVA-DL** on the
Oxford spike-train dataset, exported to NIR, and run through `snn-mlir` all the way to a
compiled binary.

> Reference model & training: LAVA-DL SLAYER Oxford tutorial —
> <https://lava-nc.org/lava-lib-dl/slayer/notebooks/oxford/train.html>

## Network structure

```
Linear(200 → 256) → CubaLIF(256) → Linear(256 → 200) → CubaLIF(200)
```

Exported as `examples/snn_oxford/network.nir`. Two `Affine`/`Linear` synapse layers feeding two
`CubaLIF` neuron populations, ending on a spike-output layer.

## Files in the example

| File | Role |
|---|---|
| `network.nir` | The trained network in NIR — the input to `snn-mlir`. |
| `input.h` | Pre-baked input spike trains (`L0_input[N_STEPS][INPUT_SIZE]`), used by `main.c`. |
| `target.csv` | Reference output (one row per timestep) for comparing against the compiled run. |
| `run.py` | Driver: exports `network.mlir` and generates the C runtime into `build/`. |
| `build/` | Generated artefacts (after running `run.py`): `network.mlir`, `snn_data.h/.c`, `main.c`, `input.h`. |

## Run it

### 1. Generate MLIR + C runtime files

```bash
uv run python examples/snn_oxford/run.py              # float32
uv run python examples/snn_oxford/run.py --quantize   # int8 weights, Q12 state
uv run python examples/snn_oxford/run.py --n-steps 50 # fewer timesteps (default 100)
```

`run.py` options:

| Flag | Default | Effect |
|---|---|---|
| `--quantize` | off | int8 weights + Q12 fixed-point neuron state |
| `--n-steps N` | `100` | Number of simulation timesteps baked into `main.c` |

This writes `examples/snn_oxford/build/`:

```
network.mlir   ← SNN dialect IR (weights baked in as constant globals)
snn_data.h     ← layer-size constants
main.c         ← memref descriptors + timestep loop + CSV output
input.h        ← copied input data
```

### 2. Lower and compile (requires the full toolchain)

```bash
export MLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir
bash pipelines/lower_cpu_linux.sh examples/snn_oxford/build/network.mlir
# → examples/snn_oxford/build/network.ll

clang examples/snn_oxford/build/network.ll \
      examples/snn_oxford/build/main.c \
      -o examples/snn_oxford/build/sim

./examples/snn_oxford/build/sim          # prints one CSV row per timestep
```

### 3. Compare against the reference

The binary prints a CSV row per timestep; compare it against `target.csv` to confirm the
compiled network reproduces the reference output (exactly in float mode; within the expected
quantization tolerance in `--quantize` mode).

<!-- TODO: if you ship a compare script, document it here -->
