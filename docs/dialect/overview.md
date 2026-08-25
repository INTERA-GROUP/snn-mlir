# The SNN MLIR dialect

## What is an MLIR dialect?

[MLIR](https://mlir.llvm.org/) is a compiler infrastructure built around the idea of
**dialects**: self-contained sets of operations, types, and attributes that model a specific
domain at a specific level of abstraction. A program is progressively *lowered* from
higher-level dialects to lower-level ones until it reaches something a machine can execute.
Crucially, every dialect's operations are first-class IR — they can be inspected, verified,
and rewritten by standard MLIR passes.

## Why a dialect for neuromorphics?

Spiking networks have semantics that general-purpose tensor IRs don't capture well: stateful
neurons that integrate over time, thresholds and resets, leak dynamics, and a sharp
distinction between spike (binary) and membrane-potential (continuous) outputs. Encoding those
directly as low-level loops loses the structure a compiler could otherwise exploit.

The `snn` dialect makes these concepts **explicit operations**. A `cubalif` neuron layer is one
op carrying its decay constants and threshold — not an opaque nest of arithmetic. That keeps
the network analyzable and retargetable: the same `network.mlir` can be lowered to a CPU today
and to custom hardware tomorrow, reusing the rest of the MLIR ecosystem.

## Operations

| Op | States | Output | Summary |
|---|---|---|---|
| `snn.linear` | — | `f32`/`i32` | Matrix-vector synapse layer (`weights @ input → output`) |
| `snn.conv2d` | — | `f32`/`i32` | 2-D convolutional synapse over `[channels, height, width]` maps |
| `snn.conv1d` | — | `f32` | 1-D convolutional synapse (float only — no quantized 1-D conv yet) |
| `snn.sumpool2d` | — | `f32`/`i32` | 2-D sum pooling |
| `snn.avgpool2d` | — | `f32`/`i32` | 2-D average pooling |
| `snn.rescale` | — | `i32` | Per-edge requantization shift to align quantization scales |
| `snn.cubalif` | current, voltage | `f32`/`i8` | Current-based leaky integrate-and-fire: two-state dynamics with threshold and voltage reset |
| `snn.cubali` | current, voltage | `f32`/`i32` | Current-based leaky integrator: two-state dynamics, continuous voltage output (no threshold) |
| `snn.lif` | voltage | `f32`/`i8` | Leaky integrate-and-fire: single-state dynamics with threshold and voltage reset |
| `snn.li` | voltage | `f32`/`i32` | Leaky integrator: single-state dynamics, continuous voltage output (no threshold) |

All ops are `memref`-based and carry explicit type information, making them directly
inspectable and transformable by standard MLIR passes.

Spike-output ops (`snn.cubalif`, `snn.lif`) emit binary activations (`f32` 0/1 or `i8` 0/1).
Voltage-output ops (`snn.cubali`, `snn.li`) emit continuous membrane potential and are used as
the final layer in regression or readout networks.

## Op interfaces

The dialect declares two native op interfaces so a consumer — a lowering, an analysis, a backend —
can read a layer **uniformly**, without switching on the concrete op type:

- **`SynapseOpInterface`** — implemented by `snn.linear`. Exposes the activation, weight,
  accumulator, and optional bias operands plus the `K`/`N` shape of `accumulator = weights @ input`.
  The convolutional and pooling ops (`snn.conv2d`/`snn.conv1d`/`snn.sumpool2d`/`snn.avgpool2d`) are
  standalone — their spatial semantics don't fit the matrix-vector contract, so they do not
  implement this interface.
- **`NeuronOpInterface`** — implemented by all four neuron ops (`snn.cubalif` / `snn.lif` /
  `snn.li` / `snn.cubali`). Exposes their operands, fixed-point parameters, and two capability
  predicates (`hasCurrentStage()`, `producesSpike()`) that distinguish the four kinds — so one
  pattern can lower every neuron.

Because they are declared natively on the ops, any `snn.linear` *is* a `SynapseOpInterface` and any
neuron op *is* a `NeuronOpInterface` — no attachment or registration. See
[Implementing a new lowering pass](lowering-pass.md#reading-a-layer-through-op-interfaces) for how
to match on them.

## Notes & design decisions

!!! note "Type-polymorphic: float **and** integer"
    Every op is designed to work with both `f32` and quantized integer types through the *same*
    operation — `f32` for simulation and faithful reference, `i8`/`i32` for quantized inference
    on **digital embedded systems**. There is no parallel "quantized op set" to keep in sync.
    How the integer parameters are computed is covered in [Quantization](../python/quantization.md).

!!! warning "`snn.rescale` is inserted automatically in quantized mode"
    When `quantize=True`, an `snn.rescale` op is inserted between `snn.linear` and the following
    neuron op to align their two quantization scales (the synapse's `w_scale` and the neuron's
    `d_scale`). It has **no NIR equivalent** — the Python export layer creates it. You will see
    it in quantized `network.mlir` output but never in float output.

!!! note "`snn.linear` covers both Linear and Affine"
    `snn.linear` handles synapse layers **with and without bias** (NIR's `Linear` and `Affine`).
    MLIR already provides an `affine` dialect, and to avoid any confusion with that unrelated
    abstraction we deliberately kept a single, neuromorphics-specific `snn.linear` op rather
    than reusing or merging with `affine`.

!!! note "Activations are rank-agnostic"
    Dense ops (`snn.linear`, the neurons) operate on 1-D activation vectors
    (`memref<Nxf32>` / `memref<Nxi32>`); the convolutional and pooling ops operate on N-D feature
    maps (`memref<CxHxWx…>`). The verifiers enforce each op's expected rank, so a shape mismatch
    is a clear error rather than a silent miscompilation. What is not supported is a **batch**
    axis — one sample per call. See [Limitations](../limitations.md).
