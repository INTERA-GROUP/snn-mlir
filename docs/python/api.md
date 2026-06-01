# API reference

The public surface of the `snn_mlir` package is intentionally small: two functions that turn a
NIR graph into SNN dialect MLIR text. Everything else (graph walking, quantization, emission)
is internal and reachable through these.

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

!!! note "Generating the C runtime is separate"
    Producing `snn_data.h/.c` and `main.c` is **not** part of the installable package — it's
    handled by the example-only helper `examples/_codegen.py`. See
    [How it works](../getting-started/how-it-works.md) and the [examples](../examples/snn-oxford.md).

<!-- When ready to auto-generate this page from docstrings, switch to the mkdocstrings plugin. -->
