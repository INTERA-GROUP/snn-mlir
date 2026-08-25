# Limitations

The current implementation covers **feedforward and recurrent** SNN topologies with
multi-dimensional dense and convolutional synapses. The constraints below are not dead ends —
they're the most useful places to contribute. Each one maps to a concrete extension point, and
we'd be glad to help you tackle any of them (see [Contributing](contributing.md)).

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
    allocates the recurrent state buffers itself, so recurrent models compile and run on the
    host like feedforward ones.

!!! warning "Convolution is float-only for 1-D; a few NIR nodes remain unmapped"
    `nir.Conv2d`, `nir.SumPool2d`, and `nir.AvgPool2d` are supported in both float and quantized
    modes; `nir.Conv1d` is supported in **float only** — there is no quantized 1-D convolution op
    yet, so a `-q` run on a `Conv1d` model is rejected. The still-unmapped NIR primitives are
    `nir.Delay`, `nir.Scale`, and `nir.Threshold`. Adding one is a contained task — see
    [Adding a NIR node type](python/nir-node.md).

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

!!! warning "No batching — one sample per call"
    Activations are now **rank-agnostic**: dense layers work on 1-D vectors and the
    convolutional/pooling ops on N-D feature maps (`[channels, height, width]`), all verified.
    What is *not* supported is **batching** — each call to the compiled function processes one
    input sample (one timestep). A leading batch axis would be a further rank increase on every
    op, and no shipped model needs it yet.

!!! warning "One reference backend (CPU)"
    The only shipped lowering is `SNNToLinalg` (CPU via `linalg`/`arith`). Additional
    targets — FPGA, ASIC, other accelerators — are exactly what the
    [lowering-pass extension point](dialect/lowering-pass.md) is for.

---

Want one of these lifted for your use case? **[We'd love your help](contributing.md)** — or
send us the NIR graph and we'll take a look.
