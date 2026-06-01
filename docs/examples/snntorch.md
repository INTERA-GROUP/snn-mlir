# SNNTorch

A second end-to-end example, exported from **[snnTorch](https://github.com/jeshraghian/snntorch/)**.
Where the [Oxford example](snn-oxford.md) is a real trained model, this one is a **decoy
network** built to exercise the pipeline: it deliberately mixes different node types and is
driven by **random inputs**, so it's useful for testing the NIR→MLIR path and the toolchain
rather than for measuring task accuracy.

## Network structure

```
Linear(784 → 256) → LIF(256) → Affine(256 → 10) → CubaLIF(10)
```

The mix is the point. In a single graph it exercises:

- **`Linear`** (`fc1`, no bias) **and** **`Affine`** (`fc2`, with bias) — both map to `snn.linear`,
  covering the bias / no-bias paths;
- a single-state **`LIF`** neuron (`lif1`) **and** a two-state **`CubaLIF`** neuron (`lif2`) —
  exercising both neuron families and, in quantized mode, two `snn.rescale` insertions.

The 784-wide input matches a flattened 28×28 frame; the random `input.h` simply drives spikes
through this topology.

## Files in the example

| File | Role |
|---|---|
| `network.nir` | The snnTorch-exported network in NIR. |
| `input.h` | Pre-baked (random) input (`L0_input[N_STEPS][INPUT_SIZE]`). |
| `target.csv` | Reference output for comparison. |
| `run.py` | Driver: exports `network.mlir` and generates the C runtime into `build/`. |

## Run it

```bash
uv run python examples/snntorch/run.py              # float32
uv run python examples/snntorch/run.py --quantize   # int8 weights, Q12 state
uv run python examples/snntorch/run.py --n-steps 50 # default is 25
```

`run.py` options:

| Flag | Default | Effect |
|---|---|---|
| `--quantize` | off | int8 weights + Q12 fixed-point neuron state |
| `--n-steps N` | `25` | Number of simulation timesteps baked into `main.c` |

The generated `build/` directory and the lower→compile→run steps are identical to the
[Oxford example](snn-oxford.md#2-lower-and-compile-requires-the-full-toolchain) — just swap the
paths from `examples/snn_oxford/` to `examples/snntorch/`.
