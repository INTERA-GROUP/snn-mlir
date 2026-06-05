// Copyright 2026 N Vision Systems And Technologies SL
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Test: op verifiers reject malformed IR with a clear diagnostic.
// Float mode must be type-uniform; quantized mode is locked to the i8/i32
// contract that the quantizer and downstream backends depend on.
// RUN: %snn-opt --split-input-file --verify-diagnostics %s

// Quantized linear must use i8 weights, not i16.
func.func @linear_bad_weight_type(
    %input:   memref<32xi8>,
    %weights: memref<64x32xi16>,
    %out:     memref<64xi32>
) {
  // expected-error @+1 {{quantized mode requires i8 weights}}
  snn.linear ins(%input, %weights) out(%out) {w_scale = 7 : i64}
      : memref<32xi8>, memref<64x32xi16> -> memref<64xi32>
  return
}

// -----

// Float linear with a mismatched output element type.
func.func @linear_float_type_mismatch(
    %input:   memref<32xf32>,
    %weights: memref<64x32xf32>,
    %out:     memref<64xf64>
) {
  // expected-error @+1 {{float mode requires input, weights, and output to share the same float element type}}
  snn.linear ins(%input, %weights) out(%out)
      : memref<32xf32>, memref<64x32xf32> -> memref<64xf64>
  return
}

// -----

// Linear weights inner dim must match the input size.
func.func @linear_shape_mismatch(
    %input:   memref<32xf32>,
    %weights: memref<64x16xf32>,
    %out:     memref<64xf32>
) {
  // expected-error @+1 {{weights inner dim (16) must match input size (32)}}
  snn.linear ins(%input, %weights) out(%out)
      : memref<32xf32>, memref<64x16xf32> -> memref<64xf32>
  return
}

// -----

// Spiking CubaLIF must emit i8 spikes in quantized mode, not i16.
func.func @cubalif_bad_output(
    %input:   memref<64xi32>,
    %current: memref<64xi32>,
    %voltage: memref<64xi32>,
    %output:  memref<64xi16>
) {
  // expected-error @+1 {{quantized mode requires i8 spike output}}
  snn.cubalif ins(%input) state(%current, %voltage) out(%output)
      {d_scale = 12 : i64, cur_decay_int = 3686 : i64,
       vol_decay_int = 4055 : i64, threshold_int = 4096 : i64}
      : memref<64xi32>, memref<64xi32>, memref<64xi32> -> memref<64xi16>
  return
}

// -----

// Voltage-readout LI must emit i32, not an i8 spike.
func.func @li_bad_output(
    %input:   memref<64xi32>,
    %voltage: memref<64xi32>,
    %output:  memref<64xi8>
) {
  // expected-error @+1 {{quantized mode requires i32 voltage output}}
  snn.li ins(%input) state(%voltage) out(%output)
      {d_scale = 12 : i64, decay_int = 3891 : i64}
      : memref<64xi32>, memref<64xi32> -> memref<64xi8>
  return
}

// -----

// Quantized state must be i32, not i16.
func.func @cubalif_bad_state(
    %input:   memref<64xi32>,
    %current: memref<64xi16>,
    %voltage: memref<64xi32>,
    %output:  memref<64xi8>
) {
  // expected-error @+1 {{quantized mode requires i32 state}}
  snn.cubalif ins(%input) state(%current, %voltage) out(%output)
      {d_scale = 12 : i64, cur_decay_int = 3686 : i64,
       vol_decay_int = 4055 : i64, threshold_int = 4096 : i64}
      : memref<64xi32>, memref<64xi16>, memref<64xi32> -> memref<64xi8>
  return
}
