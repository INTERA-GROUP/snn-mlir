# Limitations

The current implementation covers **feedforward, fully-connected** SNN topologies. The
constraints below are not dead ends — they're the most useful places to contribute. Each one
maps to a concrete extension point, and we'd be glad to help you tackle any of them (see
[Contributing](contributing.md)).

We group them by which half of the project they live in.

!!! tip "Which of these apply to *your* model?"
    ```bash
    snn-mlir check my_model.nir
    ```
    `check` applies every Python/NIR-frontend rule below to a specific graph and reports each
    node it affects, without converting anything. See
    [the CLI](getting-started/quickstart.md#4-the-full-cli-surface) or
    [`snn_mlir.check`](python/api.md#snn_mlirchecksource-report).

## Python / NIR frontend

!!! warning "Linear chains plus canonical self-recurrence only"
    The graph walker (`_graph.parse_graph`) accepts a single path from `input` to `output`,
    optionally with the canonical SNN self-recurrence — a neuron ⇄ recurrent-synapse loop,
    broken at the timestep boundary (see
    [Recurrence](python/nir-mapping.md#recurrence)). **Branching, residual connections, and
    any other cycle shape** are not yet supported. The C reference runtime (`codegen` / `run`)
    does not yet pass the recurrent state buffers, so it refuses recurrent models — MLIR
    emission only, for now.

!!! warning "No convolutional / pooling nodes"
    NIR nodes such as `nir.Conv2d`, `nir.AvgPool2d`, and `nir.SumPool2d` operate on
    `[channels, height, width]` feature maps and have no SNN equivalent yet. Adding them is a
    contained task — see [Adding a NIR node type](python/nir-node.md).

!!! warning "Uniform neuron parameters per layer"
    All neurons in a layer share the same decay constants and threshold (the parser enforces
    this). **Per-neuron parameter arrays** are not yet supported.

!!! warning "Discrete-convention NIR exports only"
    The parser recovers the export timestep as `dt = tau_mem / r` and assumes the discrete
    input gain `k = w_in · dt / tau_syn` is 1, which holds when the exporter wrote
    `r = tau_mem / dt` and `w_in = tau_syn / dt`. A `CubaLIF`/`CubaLI` node with `k ≠ 1`
    (e.g. a file exported with true continuous time constants) is rejected with an error
    naming the node and the computed `k`. See
    [the discretization convention](python/nir-mapping.md#the-discretization-convention).

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
