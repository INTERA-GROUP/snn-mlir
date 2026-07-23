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

The 784-wide input matches a flattened 28×28 frame; the random `input.csv` simply drives spikes
through this topology.

## Files in the example

| File | Role |
|---|---|
| `network.nir` | The snnTorch-exported network in NIR. |
| `input.csv` | Random input: 25 rows (timesteps) × 784 columns (input channels). |
| `build/` | Generated artefacts. Created by `codegen`/`run`, not checked in. |

## Run it

```bash
uv run snn-mlir run examples/snntorch              # float32
uv run snn-mlir run examples/snntorch --quantize   # int8 weights, Q12 state
```

The number of timesteps comes from the 25 rows of `input.csv` — swap in a longer CSV and the
generated `main.c` follows. Output lands in `examples/snntorch/build/results.csv`, 10 columns
wide.

The generated `build/` directory and the by-hand lower→compile→run steps are identical to the
[Oxford example](snn-oxford.md#the-long-way-by-hand) — just swap the paths from
`examples/snn_oxford/` to `examples/snntorch/`.
