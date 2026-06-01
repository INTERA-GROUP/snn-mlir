# Quantization

## Why quantization lives here

NIR itself has **no notion of quantization** — a NIR graph carries floating-point weights and
continuous neuron parameters. A model may have been *trained* quantization-aware and then
exported to NIR in float, or trained in pure float; either way, NIR hands us floats. Turning
those into the integer representation a digital embedded target wants is therefore the
**frontend's** job, applied on the way from NIR to MLIR.

Pass `quantize=True` to [`to_mlir()` / `export()`](api.md) to enable it. In float mode
(`quantize=False`) the weights and state stay `f32` and none of the machinery below runs.

## The scheme

The quantization is a **power-of-two (shift-based) scheme** inspired by LAVA's quantization
approach. Using powers of two means every rescaling is a bit-shift rather than a
multiply-and-divide, which is cheap and exact on embedded hardware while keeping accuracy loss
low.

<!-- TODO: confirm the "based on LAVA, very low loss" framing / add measured accuracy numbers if you want them public -->

### Weights → int8

For each synapse layer, a single per-layer scale is chosen so the largest-magnitude weight
maps near the int8 range without overflow:

```
ratio   = min( |127 / max_w|, |128 / min_w| )
w_scale = floor(log2(ratio))            # power-of-two exponent
q_w     = round(w * 2^w_scale)          # stored as int8
q_bias  = round(bias * 2^w_scale)       # stored as int32 (if present)
```

`w_scale` (the exponent, not the factor) is emitted as an attribute on `snn.linear` so the
scale is recoverable downstream.

### Neuron state → Q12 fixed point

Neuron parameters (decays, threshold) are quantized to a fixed `Q12` fixed-point format — i.e.
scaled by `2^12` — and emitted as integer attributes on the neuron op:

```
decay_int     = round(decay     * 2^12)
threshold_int = round(threshold * 2^12)
```

The state arrays (`current`, `voltage`) accordingly become `i32`, and spike outputs become
`i8` (0/1).

## The rescale step (and why it's needed)

The two scales above are independent: a `snn.linear` produces values at scale `2^w_scale`, but
the following neuron op expects its input at the neuron's `2^d_scale` (= `2^12`). They almost
never match. To bridge them, the frontend inserts a synthetic **`snn.rescale`** node on every
synapse→neuron edge that simply shifts by the difference:

```
shift = d_scale − w_scale
```

This op exists only in quantized output and has **no NIR equivalent** — it is created by the
export layer once both neighboring scales are known. Conceptually it is the small "glue" that
keeps the integer pipeline numerically aligned, replacing what would be a floating-point
rescale with a single shift.

!!! note
    Because `snn.rescale` is inserted automatically, you don't write it and won't see it in
    float-mode MLIR. It appears between each `snn.linear` and its neuron op in quantized
    `network.mlir`.
