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

## Current NIR coverage & pending nodes

The supported set above covers feedforward, fully-connected networks. NIR node types that are
**not yet mapped** include the convolutional and pooling family:

- `nir.Conv1d` / `nir.Conv2d`
- `nir.AvgPool2d` / `nir.SumPool2d`
- `nir.Flatten`, and other spatial/structural nodes

<!-- TODO: confirm/extend the exact list of pending NIR nodes you want to advertise -->

Adding one is a contained task — see [Adding a NIR node type](nir-node.md). If the model you
care about uses an unsupported node, we'd be glad to help; [send us the NIR graph](../contributing.md).
