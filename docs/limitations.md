# Limitations

The current implementation covers **feedforward, fully-connected** SNN topologies. The
constraints below are not dead ends — they're the most useful places to contribute. Each one
maps to a concrete extension point, and we'd be glad to help you tackle any of them (see
[Contributing](contributing.md)).

We group them by which half of the project they live in.

## Python / NIR frontend

!!! warning "Linear-chain graphs only"
    The graph walker (`_graph.parse_graph`) follows a single path from `input` to `output`.
    **Branching, residual connections, and recurrent edges** are not yet supported.

!!! warning "No convolutional / pooling nodes"
    NIR nodes such as `nir.Conv2d`, `nir.AvgPool2d`, and `nir.SumPool2d` operate on
    `[channels, height, width]` feature maps and have no SNN equivalent yet. Adding them is a
    contained task — see [Adding a NIR node type](python/nir-node.md).

!!! warning "Uniform neuron parameters per layer"
    All neurons in a layer share the same decay constants and threshold (the parser enforces
    this). **Per-neuron parameter arrays** are not yet supported.

## MLIR dialect & lowering

!!! warning "1-D activations only"
    All ops require 1-D activation vectors — `memref<Nxf32>` or `memref<Nxi32>`. Neuron
    populations are flat arrays, not spatial maps. The verifiers enforce this, so a 2-D feature
    map yields a clear error rather than a silent miscompilation. The neuron dynamics ops are
    already rank-agnostic at the lowering level, so extending to N-D is mostly blocked on the
    convolutional synapse op above.

!!! warning "Batch size 1"
    Each call to the compiled function processes one input sample. Batched inference would need
    2-D activation memrefs, which is blocked by the 1-D constraint.

!!! warning "One reference backend (CPU)"
    The only shipped lowering is `SNNToLinalg` (CPU via `linalg`/`arith`). Additional
    targets — FPGA, ASIC, other accelerators — are exactly what the
    [lowering-pass extension point](dialect/lowering-pass.md) is for.

## CLI & reference runtime

!!! warning "`run` targets the host CPU only"
    `snn-mlir run` compiles and executes for the machine you are on (Linux/x86-64 is what we
    test); `--platform` is reserved but accepts only `linux`. Cross-compiling is not automated —
    take the `codegen` output and lower `network.mlir` with your own target triple.

!!! warning "One model per folder, no reference comparison"
    A model folder is exactly one `.nir` plus one `input.csv`. The run writes `results.csv` and
    stops: no golden output is shipped and no accuracy check is performed, deliberately — how
    to compare against your simulator is your call.

---

Want one of these lifted for your use case? **[We'd love your help](contributing.md)** — or
send us the NIR graph and we'll take a look.
