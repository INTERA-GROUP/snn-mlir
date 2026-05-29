// SPDX-License-Identifier: Apache-2.0
//===-------------- SNN op definitions ---------------------*- tablegen -*-===//
//
// Defines snn.linear, snn.rescale, snn.cubalif. Add more in the future
//
//===----------------------------------------------------------------------===//

#ifndef SNN_OPS_TD
#define SNN_OPS_TD

include "SNN/SNNDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Base class for all SNN ops
//===----------------------------------------------------------------------===//

class SNN_Op<string mnemonic, list<Trait> traits = []>
    : Op<SNN_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// snn.linear
//===----------------------------------------------------------------------===//
//
// Type-polymorphic linear (matvec) layer.
//
//   Float mode:     memref<Nxf32>, memref<MxNxf32> -> memref<Mxf32>
//   Quantized mode: memref<Nxi8>,  memref<MxNxi8>  -> memref<Mxi16> or memref<Mxi32>
//
// w_scale is only meaningful in quantized mode. In float mode it can be
// omitted (defaults to 0, lowering ignores it).

def SNN_LinearOp : SNN_Op<"linear"> {
  let summary = "Linear (matrix-vector) synapse layer";
  let description = [{
    Computes output = weights @ input (matrix-vector product).
    Supports both f32 and quantized i8 operands.
    In quantized mode, w_scale records the power-of-2 weight scale factor.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "input vector", [MemRead]>:$input,
    Arg<AnyMemRef, "weight matrix", [MemRead]>:$weights,
    Arg<AnyMemRef, "output vector", [MemWrite]>:$output,
    Arg<Optional<AnyMemRef>, "bias vector", [MemRead]>:$bias,
    DefaultValuedAttr<I64Attr, "0">:$w_scale
  );

  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $input `,` $weights `)` (`bias` `(` $bias^ `:` type($bias) `)`)?
    `out` `(` $output `)` attr-dict `:` type($input) `,` type($weights) `->` type($output)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// snn.rescale
//===----------------------------------------------------------------------===//
//
// Per-edge requantization: bridges scale gap between synapse outputs
// (2^w_scale) and neurons input (2^d_scale) via shift.
//
// This op only appears in quantized pipelines. Float pipelines skip it.

def SNN_RescaleOp : SNN_Op<"rescale"> {
  let summary = "Per-edge requantization shift";
  let description = [{
    Applies an arithmetic shift to bridge quantization scale gaps.
    shift = d_scale - w_scale. Positive → shift left, negative → shift right.
    Input is sign-extended to i32 if narrower before shifting.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "input vector", [MemRead]>:$input,
    Arg<AnyMemRef, "output vector", [MemWrite]>:$output,
    I64Attr:$w_scale,
    I64Attr:$d_scale
  );

  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $input `)` `out` `(` $output `)`
    attr-dict `:` type($input) `->` type($output)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// snn.cubalif
//===----------------------------------------------------------------------===//
//
// CubaLIF neuron dynamics. Type-polymorphic:
//
//   Float mode:     all f32, attributes are f64 (real-valued decay/threshold)
//   Quantized mode: state i32, output i8, attributes are i64 (Q12 scaled integers)
//
// In float mode d_scale is ignored (default 0).

def SNN_CubaLIFOp : SNN_Op<"cubalif"> {
  let summary = "CubaLIF spiking neuron layer";
  let description = [{
    Computes current-based leaky integrate-and-fire dynamics:
      current = cur_decay * current + input
      voltage = vol_decay * voltage + current
      spike   = voltage > threshold ? 1 : 0
      voltage = spike ? 0 : voltage
    State (current, voltage) is updated in-place.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "input current", [MemRead]>:$input,
    Arg<AnyMemRef, "synaptic current state (updated in place)", [MemRead, MemWrite]>:$current,
    Arg<AnyMemRef, "membrane voltage state (updated in place)", [MemRead, MemWrite]>:$voltage,
    Arg<AnyMemRef, "spike output", [MemWrite]>:$output,
    // Two parameter sets, each with defaults: the I64 `*_int` attributes carry
    // the Q12 quantized values, the F64 `*_float` attributes carry the real
    // ones. The lowering reads only the set matching the operand element type
    // (integer → `*_int`, float → `*_float`); the other keeps its default.
    // d_scale is the Q12 fixed-point shift and is meaningful in quantized mode
    // only (default 0 in float mode).
    DefaultValuedAttr<I64Attr,  "0">:$d_scale,
    DefaultValuedAttr<I64Attr,  "0">:$cur_decay_int,
    DefaultValuedAttr<I64Attr,  "0">:$vol_decay_int,
    DefaultValuedAttr<I64Attr,  "0">:$threshold_int,
    DefaultValuedAttr<F64Attr,  "0.0">:$cur_decay_float,
    DefaultValuedAttr<F64Attr,  "0.0">:$vol_decay_float,
    DefaultValuedAttr<F64Attr,  "0.0">:$threshold_float
  );

  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $input `)` `state` `(` $current `,` $voltage `)` `out` `(` $output `)`
    attr-dict `:` type($input) `,` type($current) `,` type($voltage) `->` type($output)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// snn.lif
//===----------------------------------------------------------------------===//
//
// LIF (Leaky Integrate-and-Fire) neuron. Covers both LIF (decay < 1) and
// IF (decay == 1) by parameter value. Type-polymorphic:
//
//   Float mode:     state/output f32, attributes are f64
//   Quantized mode: input/state i32, output i8, attributes are i64 (Q12)
//
// v_reset defaults to 0 in both modes.

def SNN_LIFOp : SNN_Op<"lif"> {
  let summary = "LIF spiking neuron layer";
  let description = [{
    Leaky integrate-and-fire dynamics:
      voltage = decay * voltage + input
      spike   = voltage > threshold ? 1 : 0
      voltage = spike ? v_reset : voltage
    State (voltage) is updated in-place. Output carries the spike.
    IF mode: pass decay=1.0 (float) or decay_int=(1<<d_scale) (quantized).
  }];

  let arguments = (ins
    Arg<AnyMemRef, "input current", [MemRead]>:$input,
    Arg<AnyMemRef, "membrane voltage state (updated in place)", [MemRead, MemWrite]>:$voltage,
    Arg<AnyMemRef, "spike output", [MemWrite]>:$output,
    DefaultValuedAttr<I64Attr,  "0">:$d_scale,
    DefaultValuedAttr<I64Attr,  "0">:$decay_int,
    DefaultValuedAttr<I64Attr,  "0">:$threshold_int,
    DefaultValuedAttr<I64Attr,  "0">:$v_reset_int,
    DefaultValuedAttr<F64Attr,  "0.0">:$decay_float,
    DefaultValuedAttr<F64Attr,  "0.0">:$threshold_float,
    DefaultValuedAttr<F64Attr,  "0.0">:$v_reset_float
  );

  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $input `)` `state` `(` $voltage `)` `out` `(` $output `)`
    attr-dict `:` type($input) `,` type($voltage) `->` type($output)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// snn.li
//===----------------------------------------------------------------------===//
//
// LI (Leaky Integrator) neuron — no threshold or spike. Covers both LI and
// I (integrator, decay == 1) by parameter value. Type-polymorphic:
//
//   Float mode:     state/output f32, attributes are f64
//   Quantized mode: input/state/output i32, attributes are i64 (Q12)

def SNN_LIOp : SNN_Op<"li"> {
  let summary = "LI leaky integrator neuron layer";
  let description = [{
    Leaky integrator dynamics (no threshold or spike):
      voltage = decay * voltage + input
      output  = voltage
    State (voltage) is updated in-place and also written to output.
    I mode: pass decay=1.0 (float) or decay_int=(1<<d_scale) (quantized).
  }];

  let arguments = (ins
    Arg<AnyMemRef, "input current", [MemRead]>:$input,
    Arg<AnyMemRef, "membrane voltage state (updated in place)", [MemRead, MemWrite]>:$voltage,
    Arg<AnyMemRef, "voltage output", [MemWrite]>:$output,
    DefaultValuedAttr<I64Attr,  "0">:$d_scale,
    DefaultValuedAttr<I64Attr,  "0">:$decay_int,
    DefaultValuedAttr<F64Attr,  "0.0">:$decay_float
  );

  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $input `)` `state` `(` $voltage `)` `out` `(` $output `)`
    attr-dict `:` type($input) `,` type($voltage) `->` type($output)
  }];

  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// snn.cubali
//===----------------------------------------------------------------------===//
//
// CubaLI (Current-Based Leaky Integrator) neuron. Two states (current,
// voltage); voltage output — no threshold or spike. Type-polymorphic:
//
//   Float mode:     all f32, attributes are f64
//   Quantized mode: state/output i32, attributes are i64 (Q12)

def SNN_CubaLIOp : SNN_Op<"cubali"> {
  let summary = "CubaLI current-based leaky integrator neuron layer";
  let description = [{
    Current-based leaky integrator dynamics (no threshold or spike):
      current = cur_decay * current + input
      voltage = vol_decay * voltage + current
      output  = voltage
    State (current, voltage) is updated in-place.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "input current", [MemRead]>:$input,
    Arg<AnyMemRef, "synaptic current state (updated in place)", [MemRead, MemWrite]>:$current,
    Arg<AnyMemRef, "membrane voltage state (updated in place)", [MemRead, MemWrite]>:$voltage,
    Arg<AnyMemRef, "voltage output", [MemWrite]>:$output,
    DefaultValuedAttr<I64Attr,  "0">:$d_scale,
    DefaultValuedAttr<I64Attr,  "0">:$cur_decay_int,
    DefaultValuedAttr<I64Attr,  "0">:$vol_decay_int,
    DefaultValuedAttr<F64Attr,  "0.0">:$cur_decay_float,
    DefaultValuedAttr<F64Attr,  "0.0">:$vol_decay_float
  );

  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $input `)` `state` `(` $current `,` $voltage `)` `out` `(` $output `)`
    attr-dict `:` type($input) `,` type($current) `,` type($voltage) `->` type($output)
  }];

  let hasVerifier = 1;
}

#endif // SNN_OPS_TD
