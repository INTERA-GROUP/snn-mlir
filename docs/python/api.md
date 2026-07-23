# API reference

The public surface of the `snn_mlir` package is small and comes in three flavors:

- **One-shot** — [`to_mlir`](#snn_mlirto_mlirsource-quantizefalse-str) /
  [`export`](#snn_mlirexportsource-output_path-quantizefalse-none) turn a NIR graph straight
  into SNN dialect MLIR text.
- **Structured** — [`parse_graph`](#snn_mlirparse_graphsource-listnodeinfo),
  [`quantize_layers`](#snn_mlirquantize_layerslayers-none), and
  [`mlir_from_layers`](#snn_mlirmlir_from_layerslayers-quantizefalse-str) expose the pipeline
  one stage at a time, so you can inspect or quantize the parsed layers — or feed them to your
  own code generation — before emitting MLIR. `to_mlir` is simply these three composed.
- **Folder-level** — [`codegen_folder`](#snn_mlircodegen_folderfolder-quantizefalse-index_bits64-path)
  and [`run_folder`](#snn_mlirrun_folderfolder-quantizefalse-platformlinux-path) are what the
  `snn-mlir codegen` and `snn-mlir run` commands call. They take a model folder and produce the
  C reference runtime — and, for `run_folder`, a compiled and executed binary.

The layer objects are [`NodeInfo`](nir-node.md) instances; both `NodeInfo` and the
`NODE_PARSERS` registry are re-exported at the top level for convenience.

!!! note "Weights are baked in"
    Synapse weights (and biases) are emitted as module-level `memref.global "private" constant`
    values and read back with `memref.get_global`, rather than passed as function arguments. The
    generated `@snn_forward_step` function therefore takes only the runtime input, the carried
    neuron state, and the output buffer — the compiled module is self-contained.

```python
import snn_mlir
```

---

## `snn_mlir.to_mlir(source, *, quantize=False) -> str`

Convert a NIR graph to SNN dialect MLIR text.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `source` | `nir.NIRGraph \| str \| Path` | — | A NIR graph object, or a path to a `.nir` file (read with `nir.read`). |
| `quantize` | `bool` | `False` | If `True`, emit int8 weights and Q12 fixed-point neuron state (inserting `snn.rescale` as needed). If `False`, emit `f32`. |

**Returns** — `str`: the complete MLIR module, ready to pipe into `snn-opt`.

**Example**

```python
import snn_mlir

mlir = snn_mlir.to_mlir("network.nir", quantize=True)
with open("network.mlir", "w") as f:
    f.write(mlir)
```

---

## `snn_mlir.export(source, output_path, *, quantize=False) -> None`

Convert a NIR graph and write the result straight to a `.mlir` file. A thin convenience wrapper
around `to_mlir` (above); it creates parent directories as needed.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `source` | `nir.NIRGraph \| str \| Path` | — | A NIR graph object, or a path to a `.nir` file. |
| `output_path` | `str \| Path` | — | Destination `.mlir` file path. Parent dirs are created automatically. |
| `quantize` | `bool` | `False` | Passed through to `to_mlir`. |

**Returns** — `None`.

**Example**

```python
import snn_mlir

snn_mlir.export("network.nir", "build/network.mlir", quantize=True)
```

---

## `snn_mlir.parse_graph(source) -> list[NodeInfo]`

Walk a NIR graph and return its ordered list of layers, stopping after parsing. Use this when
you need the [`NodeInfo`](nir-node.md) objects themselves — to inspect them, or to drive your own
code generation — rather than just the MLIR text.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `source` | `nir.NIRGraph \| str \| Path` | — | A NIR graph object, or a path to a `.nir` file. |

**Returns** — `list[NodeInfo]`: the ordered layers, with **no** quantization applied.

---

## `snn_mlir.quantize_layers(layers) -> None`

Compute each layer's quantization parameters (int8 weight scales, Q12 neuron state) **in-place**.

!!! warning "Call at most once per list"
    Quantization is not idempotent — running it twice re-scales already-quantized weights. Parse
    a fresh list with `parse_graph` if you need to quantize again.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `layers` | `list[NodeInfo]` | — | Layers from `parse_graph`, mutated in place. |

**Returns** — `None`.

---

## `snn_mlir.mlir_from_layers(layers, *, quantize=False) -> str`

Emit SNN dialect MLIR text from a pre-parsed list of layers. In quantized mode it inserts the
synthetic `snn.rescale` nodes before emission.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `layers` | `list[NodeInfo]` | — | Layers from `parse_graph` (run through `quantize_layers` first if `quantize=True`). |
| `quantize` | `bool` | `False` | Must match how `layers` were quantized: `True` emits int8 / Q12 (with `snn.rescale`), `False` emits `f32`. |

**Returns** — `str`: the complete MLIR module.

**Example** — the structured pipeline (`to_mlir` is exactly this composed):

```python
import snn_mlir

layers = snn_mlir.parse_graph("network.nir")
snn_mlir.quantize_layers(layers)              # inspect / use `layers` here too
mlir = snn_mlir.mlir_from_layers(layers, quantize=True)
```

---

## `snn_mlir.codegen_folder(folder, *, quantize=False, index_bits=64) -> Path`

Generate the CPU reference sources for a **model folder** — the function behind
`snn-mlir codegen`. The folder must hold exactly one `*.nir` and an `input.csv`; the output goes
to `<folder>/build/`.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `folder` | `str \| Path` | — | Model folder: exactly one `*.nir` plus `input.csv`. |
| `quantize` | `bool` | `False` | int8 weights and Q12 fixed-point neuron state. |
| `index_bits` | `int` | `64` | Width of the memref descriptor index fields. Use `32` for a 32-bit embedded target. Not exposed on the CLI. |

**Returns** — `Path`: the generated `build/` directory, containing `network.mlir`,
`snn_data.h`, `input.h`, and `main.c`.

**Raises** — `FileNotFoundError` if the folder, the `.nir`, or `input.csv` is missing;
`ValueError` if the folder holds more than one `.nir`, or if `input.csv`'s column count does not
match the network's input size.

!!! note "The CSV sets the timestep count"
    `n_steps` is the **row count** of `input.csv`, and each row is baked into `input.h` as
    `int8_t L0_input[N_STEPS][INPUT_SIZE]`. There is no `n_steps` parameter.

**Example**

```python
import snn_mlir

build = snn_mlir.codegen_folder("my_model", quantize=True, index_bits=32)
print(sorted(p.name for p in build.iterdir()))
```

---

## `snn_mlir.run_folder(folder, *, quantize=False, platform="linux") -> Path`

Codegen, compile, and execute a model folder — the function behind `snn-mlir run`. Lowers
`network.mlir` through `snn-opt` and `mlir-opt`, compiles it with `llc`, links the generated
`main.c` against it with the system C compiler, runs the binary, and captures its per-timestep
output.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `folder` | `str \| Path` | — | Model folder, as for `codegen_folder`. |
| `quantize` | `bool` | `False` | int8 weights and Q12 fixed-point neuron state. |
| `platform` | `str` | `"linux"` | Reserved for future targets; only `"linux"` is accepted today. |

**Returns** — `Path`: the written `build/results.csv`, one row per timestep. No reference
comparison is performed.

**Raises** — `FileNotFoundError` (with a per-tool breakdown) if the toolchain is incomplete;
`subprocess.CalledProcessError` if a tool fails; `ValueError` for an unsupported `platform`.
Every intermediate — `network.ll`, `network.o`, the executable — is left on disk.

**Example**

```python
import snn_mlir

if snn_mlir.toolchain_available():
    results = snn_mlir.run_folder("my_model", quantize=True)
    print(results.read_text())
```

---

## `snn_mlir.toolchain_available() -> bool`

`True` if every tool `run_folder` needs (`snn-opt`, `mlir-opt`, `mlir-translate`, `llc`, and a C
compiler) can be resolved. Use it to skip tests or degrade gracefully instead of catching the
`FileNotFoundError`. Resolution order and the `SNN_OPT` / `MLIR_DIR` / `CC` overrides are
documented in [Installation](../getting-started/installation.md).

<!-- When ready to auto-generate this page from docstrings, switch to the mkdocstrings plugin. -->
