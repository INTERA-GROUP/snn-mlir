# NIR mapping

## What NIR is, and why we support it

The [Neuromorphic Intermediate Representation (NIR)](https://neuroir.org/) is a
framework-neutral format for describing spiking and continuous neuron networks as a graph of
well-defined node types (`Linear`, `Affine`, `LIF`, `CubaLIF`, …). It is the lingua franca that
lets a model trained in one framework be read by another. By consuming NIR, `snn-mlir` becomes
**framework-agnostic** for free: any of the [supported simulators](../index.md#supported-simulators)
that export NIR can target the dialect, instead of us writing a bespoke importer per framework.

!!! note "NIR is broader than MLIR's digital world"
    NIR is designed to describe **both digital and analog** neuron models, using continuous
    physical parameters — membrane and synaptic time constants (`tau_mem`, `tau_syn`),
    resistance (`r`), leak (`v_leak`), threshold (`v_threshold`), and so on. The `snn` dialect,
    by contrast, targets **digital, discrete-time** execution. The frontend therefore
    *discretizes* the continuous NIR parameters into the per-timestep decay factors the dialect
    uses.

## Discretization

For each neuron node, the parser derives the discrete-time update factors from the NIR physical
parameters. For a `CubaLIF` node, for example:

```
dt        = tau_mem / r
cur_decay = 1 − dt / tau_syn      # current (synaptic) leak per step
vol_decay = 1 − dt / tau_mem      # voltage (membrane) leak per step
threshold = v_threshold
```

Integrate-and-fire variants (`CubaIF`, `IF`) are simply the leaky case with the decay set to
`1.0`, which disables the exponential leak. The parser also enforces the dialect's
assumptions — e.g. `v_leak` must be 0, and `tau_syn`, `tau_mem`, `v_threshold` must be uniform
across the layer (see [Limitations](../limitations.md)).

### The discretization convention

NIR is deliberately continuous — it describes neuron dynamics with physical time constants and
"abstracts away" the simulator's timestep. A discrete-time exporter therefore has to smuggle its
timestep into the file somewhere, and the ecosystem convention is to use `r` and `w_in` as the
carriers:

```
r    = tau_mem / dt        # so the parser can recover dt = tau_mem / r
w_in = tau_syn / dt        # so the discrete input gain k = w_in · dt / tau_syn = 1
```

The frontend relies on this. By the `dt = tau_mem / r` construction, the `LIF`/`LI` input gain
and the `CubaLIF`/`CubaLI` *voltage*-stage gain are identically 1 for any `r`; the one remaining
free parameter is the *current*-stage gain `k = w_in · dt / tau_syn`, and the emitted update
`current += input` assumes `k = 1`. The parser checks exactly that: a `CubaLIF`/`CubaLI` node
whose parameters give `k ≠ 1` (within float32 tolerance) is **rejected** with an error naming
the node and the computed `k`, rather than silently compiling dynamics different from the ones
trained.

Files written with true continuous time constants (e.g. `r = 1`, `w_in = 1`) will fail this
check. That is intentional for now: folding a `k ≠ 1` gain into the weights or threshold is a
planned ingestion feature, and until it lands, rejecting is the honest alternative to
miscompiling.

## Node mapping

Each SNN op covers a family of NIR nodes:

| NIR node | SNN op | Notes |
|---|---|---|
| `nir.Linear` | `snn.linear` | No bias |
| `nir.Affine` | `snn.linear` | Bias added as second operand |
| `nir.CubaLIF` | `snn.cubalif` | `cur_decay`, `vol_decay` < 1 |
| `nir.CubaIF` | `snn.cubalif` | `cur_decay = vol_decay = 1.0` (no leak) |
| `nir.CubaLI` | `snn.cubali` | `cur_decay`, `vol_decay` < 1 |
| `nir.CubaI` | `snn.cubali` | `cur_decay = vol_decay = 1.0` (no leak) |
| `nir.LIF` | `snn.lif` | `decay` < 1 |
| `nir.IF` | `snn.lif` | `decay = 1.0` (no leak) |
| `nir.LI` | `snn.li` | `decay` < 1 |
| `nir.I` | `snn.li` | `decay = 1.0` (no leak) |
| _(internal)_ | `snn.rescale` | Inserted between `snn.linear` and neuron ops during quantized export; no NIR equivalent |

## Recurrence

The one supported cycle is the canonical SNN self-recurrence: a neuron whose spikes feed a
recurrent synapse that projects back onto the same neuron —

```
input → fc1 → lif1 → fc2 → …
         ↑      ↓
       w_rec ← ─┘        (lif1 → w_rec → lif1)
```

The graph walk breaks the **neuron→synapse** edge (`lif1 → w_rec`) at the timestep boundary:
`w_rec` reads the *previous* timestep's spikes from a dedicated state buffer, which keeps that
buffer spike-typed (`i8` quantized / `f32` float). This is the standard discrete-time SNN
reading of a recurrent connection. Any other cycle — a self-loop, a longer loop, a loop without
a neuron→synapse edge — is rejected with an error.

Three consequences in the emitted MLIR:

1. **Execution order.** The recurrent synapse runs *first* in the timestep (it depends only on
   stored state), so for the graph above the order is `w_rec, fc1, lif1, fc2, …` —
   `GraphInfo.order` is the authority.
2. **Fan-in merge.** The neuron now has two inputs (`fc1` and `w_rec`). Each edge gets its own
   `snn.rescale` in quantized mode, and the branches are summed elementwise by a
   `linalg.generic` (`arith.addi` in Q12; `arith.addf` in float) just before the neuron.
   Synapses feeding the same neuron are clamped to one shared `w_scale` (the minimum of the
   candidates) — a uniform fan-in scale is what lets any backend fuse the merge without
   changing the result, since the rescale's left shift distributes over the addition.
3. **State buffer + copy.** The previous-spikes buffer is a function argument
   (`%prev_spikes_<neuron>`), and the function ends with a `memref.copy` of the neuron's fresh
   spikes into it for the next call.

### The argument-order contract

The generated `@snn_forward_step` is called positionally from C, so its argument order is an
ABI. It is, deterministically:

1. `%input`
2. for each layer in `GraphInfo.order`: its state buffers (e.g. `%current_<n>`,
   `%voltage_<n>`), and — immediately after them, when the layer is a recurrent neuron — its
   previous-spikes buffer `%prev_spikes_<n>`
3. `%output`

Any `main.c`-style caller must mirror this exactly; a mismatch is silent memory corruption, not
a link error.

## Current NIR coverage & pending nodes

The supported set above covers feedforward, fully-connected networks. NIR node types that are
**not yet mapped** include the convolutional and pooling family:

- `nir.Conv1d` / `nir.Conv2d`
- `nir.AvgPool2d` / `nir.SumPool2d`
- `nir.Flatten`, and other spatial/structural nodes

<!-- TODO: confirm/extend the exact list of pending NIR nodes you want to advertise -->

Adding one is a contained task — see [Adding a NIR node type](nir-node.md). If the model you
care about uses an unsupported node, we'd be glad to help; [send us the NIR graph](../contributing.md).
