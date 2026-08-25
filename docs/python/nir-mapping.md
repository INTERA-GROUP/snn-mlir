# NIR mapping

## What NIR is, and why we support it

The [Neuromorphic Intermediate Representation (NIR)](https://neuroir.org/) is a
framework-neutral format for describing spiking and continuous neuron networks as a graph of
well-defined node types (`Linear`, `Affine`, `LIF`, `CubaLIF`, …). It is the lingua franca that
lets a model trained in one framework be read by another. By consuming NIR, `snn-mlir` becomes
**framework-agnostic** for free: any simulator that exports the
[supported NIR nodes](../index.md#supported-nir-nodes) can target the dialect, instead of us
writing a bespoke importer per framework.

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

The parser also enforces the dialect's assumptions — e.g. `v_leak` must be 0, and `tau_syn`,
`tau_mem`, `v_threshold` must be uniform across the layer (see
[Limitations](../limitations.md)).

### The non-leaky nodes are not the leaky ones with `decay = 1`

`nir.IF` and `nir.I` do describe the same dynamics as `LIF`/`LI` with the leak disabled, but that
is a statement about the *neuron*, not a recipe for the *parser*. They carry **no `tau` and no
`v_leak` field at all**, so:

* there is nothing to discretize — `decay = 1` is the definition of the node, not a computed
  result — and the leaky parser would simply fail on the missing `tau`; and
* **`r` changes meaning.** In `tau·dv/dt = (v_leak − v) + r·i` the exporter convention `r = tau/dt`
  makes `r` cancel out of the input gain entirely (it survives only in `decay = 1 − 1/r`), which is
  why `LIF`/`LI` accept **any** `r`. `IF`/`I` have the equation `dv/dt = r·i`: no `tau`, so nothing
  carries `dt` and nothing cancels. `r` is left as a bare gain on the input, and the update these
  emit is `voltage += input`. A node with `r ≠ 1` is therefore **rejected**, in the same spirit as
  the `CubaLIF` gain check below.

`snn_mlir` handles this with two extra *parser functions* (`parse_if`, `parse_i`) feeding the
existing `snn.lif` / `snn.li` ops — the same shape as `parse_linear`/`parse_affine`, which both
build `snn.linear`. The dialect stays at four neuron ops.

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
| `nir.Conv2d` | `snn.conv2d` | 2-D convolutional synapse; float + quantized |
| `nir.Conv1d` | `snn.conv1d` | 1-D convolutional synapse; float + quantized |
| `nir.SumPool2d` | `snn.sumpool2d` | Sum pooling; float + quantized |
| `nir.AvgPool2d` | `snn.avgpool2d` | Average pooling; float + quantized |
| `nir.Flatten` | _(reshape)_ | Structural — flattens a feature map to a vector; no dedicated op |
| `nir.CubaLIF` | `snn.cubalif` | `cur_decay`, `vol_decay` < 1 |
| `nir.CubaLI` | `snn.cubali` | `cur_decay`, `vol_decay` < 1 |
| `nir.LIF` | `snn.lif` | `decay` < 1, derived from `tau`/`r` |
| `nir.IF` | `snn.lif` | `decay = 1.0` by definition; requires `r = 1` |
| `nir.LI` | `snn.li` | `decay` < 1, derived from `tau`/`r` |
| `nir.I` | `snn.li` | `decay = 1.0` by definition; requires `r = 1` |
| _(internal)_ | `snn.rescale` | Inserted between a synapse and neuron ops during quantized export; no NIR equivalent |

NIR has no cumulative-current integrate-and-fire node — there is no `nir.CubaIF` or `nir.CubaI`
— so `snn.cubalif`/`snn.cubali` are reachable only from their leaky counterparts. Thirteen NIR
node types map today; the table above is the complete list. The four neuron families
(`CubaLIF`/`CubaLI`/`LIF`/`LI`, plus the non-leaky `IF`/`I` pair that reuses `snn.lif`/`snn.li`)
still collapse to just **four neuron ops** in the dialect.

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

Because the ABI is positional, names never cross it — which matters for NIR node names that are
not valid C identifiers (dotted submodule paths like `lif1.lif` are common). MLIR keeps the
name verbatim; C generators use `NodeInfo.c_name`, the name with every non-identifier character
replaced by `_` (`lif1.lif` → `lif1_lif`). Since that mangling can collide (`a.b` and `a_b`),
`codegen` checks the graph's C names are unique up front and fails loudly if not.

## Current NIR coverage & pending nodes

The supported set above covers dense and convolutional feedforward networks, plus canonical SNN
self-recurrence (see [Recurrence](#recurrence)). The NIR primitives **not yet mapped** are:

- `nir.Delay` — a fixed time-delay edge
- `nir.Scale` — a scalar gain node
- `nir.Threshold` — a standalone threshold node

Both convolutions map in float **and** quantized mode. LLVM ships no quantized rank-3 named
conv, so `nir.Conv1d` is lowered by embedding it in a 2-D convolution with a unit width axis and
reusing `linalg.conv_2d_nchw_fchw_q` — exact, since the extra axis is size 1. Branching and
residual topologies remain unsupported (only linear chains and the one recurrent cycle above).

Adding a node is a contained task — see [Adding a NIR node type](nir-node.md). If the model you
care about uses an unsupported node, we'd be glad to help; [send us the NIR graph](../contributing.md).
