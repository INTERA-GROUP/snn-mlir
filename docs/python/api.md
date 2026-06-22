# API reference

The public surface of the `snn_mlir` package is small and comes in two flavors:

- **One-shot** — [`to_mlir`](#snn_mlirto_mlirsource-quantizefalse-str) /
  [`export`](#snn_mlirexportsource-output_path-quantizefalse-none) turn a NIR graph straight
  into SNN dialect MLIR text.
- **Structured** — [`parse_graph`](#snn_mlirparse_graphsource-listnodeinfo),
  [`quantize_layers`](#snn_mlirquantize_layerslayers-none), and
  [`mlir_from_layers`](#snn_mlirmlir_from_layerslayers-quantizefalse-str) expose the pipeline
  one stage at a time, so you can inspect or quantize the parsed layers — or feed them to your
  own code generation — before emitting MLIR. `to_mlir` is simply these three composed.

The layer objects are [`NodeInfo`](nir-node.md) instances; both `NodeInfo` and the
`NODE_PARSERS` registry are re-exported at the top level for convenience.

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

!!! note "Generating the C runtime is separate"
    Producing `snn_data.h/.c` and `main.c` is **not** part of the installable package — it's
    handled by the example-only helper `examples/_codegen.py`. See
    [How it works](../getting-started/how-it-works.md) and the [examples](../examples/snn-oxford.md).

<!-- When ready to auto-generate this page from docstrings, switch to the mkdocstrings plugin. -->
