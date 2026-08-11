# API reference

The public surface of the `snn_mlir` package is small and comes in four flavors:

- **Checking** — [`check`](#snn_mlirchecksource-report) answers whether a graph is supported
  *before* converting it, reporting every node rather than raising on the first problem.
- **One-shot** — [`to_mlir`](#snn_mlirto_mlirsource-quantizefalse-str) /
  [`export`](#snn_mlirexportsource-output_path-quantizefalse-none) turn a NIR graph straight
  into SNN dialect MLIR text.
- **Structured** — [`parse_graph`](#snn_mlirparse_graphsource-graphinfo),
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
    neuron state (including, for a recurrent neuron, its previous-timestep spike buffer — see
    [Recurrence](nir-mapping.md#recurrence)), and the output buffer — the compiled module is
    self-contained.

```python
import snn_mlir
```

---

## `snn_mlir.check(source) -> Report`

Report whether a NIR graph can be converted, and what blocks it.

Where `parse_graph` raises on the first thing it cannot handle, `check` runs the same rules over
every node independently and collects the results. Node-level rules are not reimplemented: each
node is handed to its real parser from `NODE_PARSERS`, and the exception it raises *is* the
finding — so the report can never drift from the parser that produced it. Only the topology walk
is restated, because it is structural rather than semantic.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `source` | `nir.NIRGraph \| str \| Path` | — | A NIR graph object, or a path to a `.nir` file (read with `nir.read`). |

**Returns** — [`Report`](#report)

!!! note "Never raises for a graph"
    `check` is total: any graph produces a report, cyclic edge sets included. Passing a *path*
    still reads it with `nir.read`, which may raise if the file is missing or malformed.

**Example**

```python
import snn_mlir

report = snn_mlir.check("network.nir")
if not report.ok:
    for f in report.errors:
        print(f"{f.node or 'graph'}: {f.message}")
```

### `Report`

| Attribute | Type | Description |
|---|---|---|
| `ok` | `bool` | `True` when nothing of severity `error` was found — i.e. `parse_graph` would succeed. |
| `nodes` | `tuple[NodeReport, ...]` | One entry per graph node, in the graph's own node order. |
| `graph` | `tuple[Finding, ...]` | Whole-graph findings: topology, missing terminals, reachability. |
| `order` | `tuple[str, ...]` | Node names along the `input` → `output` path, in data-flow order. Empty when the walk could not complete. |
| `findings` | `list[Finding]` | Every finding, node-level first. |
| `errors` / `warnings` | `list[Finding]` | `findings` split by severity. |
| `as_dict()` | `dict` | JSON-serializable form, as emitted by `snn-mlir check --json`. |

### `NodeReport`

| Attribute | Type | Description |
|---|---|---|
| `name` | `str` | Node name as it appears in the graph. |
| `type` | `str` | NIR node class name, e.g. `CubaLIF`. |
| `role` | `str` | `terminal`, `synapse`, `neuron`, or `unsupported`. |
| `ok` | `bool` | Whether this node alone can be parsed. |
| `findings` | `tuple[Finding, ...]` | Why not, when `ok` is `False`. |

### `Finding`

| Attribute | Type | Description |
|---|---|---|
| `kind` | `str` | `unsupported_type`, `unsupported_parameter`, `nonlinear_topology`, `unsupported_fan_in`, `dead_end`, `cycle`, `recurrent_edge`, `missing_terminal`, `unreachable`, or `parser_error`. |
| `message` | `str` | Human-readable sentence — for node findings, verbatim what the parser raised. |
| `severity` | `str` | `error` (will not convert), `warning` (converts, worth knowing), or `info` (a structural fact the conversion handles). |
| `node` | `str \| None` | The node it applies to, or `None` for whole-graph findings. |

!!! tip "A model can be all-green per node and still unsupported"
    An unbreakable cycle is the case to keep in mind: every node parses on its own, but the
    edges cannot be ordered. That is why topology is checked separately and reported under
    `report.graph` rather than against any one node. The one cycle that *is* supported — the
    canonical neuron ⇄ recurrent-synapse loop (see
    [Recurrence](nir-mapping.md#recurrence)) — appears as an `info`-severity `recurrent_edge`
    finding anchored to the recurrent synapse, and does not affect `ok`.

!!! note "`parser_error` means a package bug, not an unsupported model"
    Findings of kind `parser_error` come from a parser raising something other than a deliberate
    rejection — a missing NIR attribute, say. They are reported rather than raised, because
    totality is the function's contract, but they are worth [reporting upstream](../contributing.md).

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

## `snn_mlir.parse_graph(source) -> GraphInfo`

Walk a NIR graph and return its layers ordered for one forward timestep, stopping after parsing.
Use this when you need the [`NodeInfo`](nir-node.md) objects themselves — to inspect them, or to
drive your own code generation — rather than just the MLIR text.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `source` | `nir.NIRGraph \| str \| Path` | — | A NIR graph object, or a path to a `.nir` file. |

**Returns** — [`GraphInfo`](#graphinfo): the parsed graph, with **no** quantization applied.

### `GraphInfo`

The layers plus the structure connecting them. Iterating, indexing, or `len()`-ing a
`GraphInfo` yields the forward-path layers in execution order, so code written against the
former `list[NodeInfo]` return keeps working unchanged.

| Attribute | Type | Description |
|---|---|---|
| `nodes` | `dict[str, NodeInfo]` | Parsed layers keyed by NIR node name (terminals excluded). |
| `order` | `list[str]` | Node names in forward (topological) execution order. A recurrent synapse comes first — it reads the previous timestep's spikes. |
| `edges` | `list[tuple[str, str]]` | Forward edges after cycle breaking, `input`/`output` terminals included. |
| `recurrent_edges` | `list[tuple[str, str]]` | Broken neuron→synapse edges: for each `(neuron, synapse)`, the synapse reads the neuron's previous-timestep spikes (see [Recurrence](nir-mapping.md#recurrence)). Empty for a feedforward chain. |
| `layers` | `list[NodeInfo]` | The forward-path layers in execution order (what iteration yields). |
| `predecessors(name)` / `successors(name)` | `list[str]` | Neighbors of a node along `edges`. |

---

## `snn_mlir.quantize_layers(layers) -> None`

Compute each layer's quantization parameters (int8 weight scales, Q12 neuron state) **in-place**.

!!! warning "Call at most once per list"
    Quantization is not idempotent — running it twice re-scales already-quantized weights. Parse
    a fresh list with `parse_graph` if you need to quantize again.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `layers` | `GraphInfo \| list[NodeInfo]` | — | Layers from `parse_graph`, mutated in place. |

**Returns** — `None`.

---

## `snn_mlir.mlir_from_layers(layers, *, quantize=False) -> str`

Emit SNN dialect MLIR text from a pre-parsed list of layers. In quantized mode it inserts the
synthetic `snn.rescale` nodes before emission.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `layers` | `GraphInfo \| list[NodeInfo]` | — | Layers from `parse_graph` (run through `quantize_layers` first if `quantize=True`). A plain list is treated as a linear chain. |
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
