// Copyright 2026 N Vision Systems And Technologies SL
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Test: op verifiers reject malformed IR with a clear diagnostic.
// Float mode must be type-uniform; quantized mode is locked to the i8/i32
// contract that the quantizer and downstream backends depend on. The neuron ops
// and snn.rescale accept any rank, but every operand must carry the same shape.
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

// -----

// Neuron operands must agree DIMENSION BY DIMENSION, not merely in element
// count. 2*3*4 == 24, so an element-count rule would accept this silently and
// hand the lowering an iteration space its state operand cannot be indexed
// with. Collapsing a feature map to a vector is memref.collapse_shape's job.
func.func @lif_state_rank_mismatch(
    %input:   memref<2x3x4xf32>,
    %voltage: memref<24xf32>,
    %output:  memref<2x3x4xf32>
) {
  // expected-error @+1 {{state operand must have the same shape as the input}}
  snn.lif ins(%input) state(%voltage) out(%output)
      {decay_float = 9.500000e-01 : f64,
       threshold_float = 1.000000e+00 : f64}
      : memref<2x3x4xf32>, memref<24xf32> -> memref<2x3x4xf32>
  return
}

// -----

// Same rank, one differing extent: still a shape disagreement.
func.func @lif_output_shape_mismatch(
    %input:   memref<2x3x4xf32>,
    %voltage: memref<2x3x4xf32>,
    %output:  memref<2x3x5xf32>
) {
  // expected-error @+1 {{expects the output to have the same shape as the input}}
  snn.lif ins(%input) state(%voltage) out(%output)
      {decay_float = 9.500000e-01 : f64,
       threshold_float = 1.000000e+00 : f64}
      : memref<2x3x4xf32>, memref<2x3x4xf32> -> memref<2x3x5xf32>
  return
}

// -----

// snn.rescale is shape-preserving too — it shifts each element in place.
func.func @rescale_shape_mismatch(
    %input:  memref<16x16x16xi32>,
    %output: memref<16x16xi32>
) {
  // expected-error @+1 {{input and output must have the same shape}}
  snn.rescale ins(%input) out(%output) {w_scale = 7 : i64, d_scale = 12 : i64}
      : memref<16x16x16xi32> -> memref<16x16xi32>
  return
}

// -----

// snn.conv2d output height must follow (H + 2p - Kh)/s + 1.
func.func @conv2d_bad_output_height(
    %input:   memref<2x34x34xf32>,
    %weights: memref<16x2x5x5xf32>,
    %output:  memref<16x17x16xf32>
) {
  // expected-error @+1 {{output height (17) does not match (H+2p-Kh)/s+1 (16)}}
  snn.conv2d ins(%input, %weights) out(%output)
      {stride = array<i64: 2, 2>, padding = array<i64: 1, 1>}
      : memref<2x34x34xf32>, memref<16x2x5x5xf32> -> memref<16x17x16xf32>
  return
}

// -----

// snn.conv2d weights in-channels must match the input channels.
func.func @conv2d_channel_mismatch(
    %input:   memref<2x16x16xf32>,
    %weights: memref<8x3x3x3xf32>,
    %output:  memref<8x14x14xf32>
) {
  // expected-error @+1 {{weights in-channels (3) must match input channels (2)}}
  snn.conv2d ins(%input, %weights) out(%output)
      : memref<2x16x16xf32>, memref<8x3x3x3xf32> -> memref<8x14x14xf32>
  return
}

// -----

// snn.conv2d in float mode is type-uniform across input, weights and output.
func.func @conv2d_float_type_mismatch(
    %input:   memref<16x16x16xf32>,
    %weights: memref<8x16x3x3xf32>,
    %output:  memref<8x14x14xf16>
) {
  // expected-error @+1 {{float mode requires input, weights, and output to share the same float element type}}
  snn.conv2d ins(%input, %weights) out(%output)
      : memref<16x16x16xf32>, memref<8x16x3x3xf32> -> memref<8x14x14xf16>
  return
}

// -----

// snn.conv1d output length must follow (L + 2p - K)/s + 1.
func.func @conv1d_bad_output_length(
    %input:   memref<2x34xf32>,
    %weights: memref<16x2x5xf32>,
    %output:  memref<16x17xf32>
) {
  // expected-error @+1 {{output length (17) does not match (L+2p-K)/s+1 (16)}}
  snn.conv1d ins(%input, %weights) out(%output)
      {stride = 2 : i64, padding = 1 : i64}
      : memref<2x34xf32>, memref<16x2x5xf32> -> memref<16x17xf32>
  return
}

// -----

// snn.conv1d weights in-channels must match the input channels.
func.func @conv1d_channel_mismatch(
    %input:   memref<2x16xf32>,
    %weights: memref<8x3x3xf32>,
    %output:  memref<8x14xf32>
) {
  // expected-error @+1 {{weights in-channels (3) must match input channels (2)}}
  snn.conv1d ins(%input, %weights) out(%output)
      : memref<2x16xf32>, memref<8x3x3xf32> -> memref<8x14xf32>
  return
}

// -----

// snn.conv1d in float mode is type-uniform across input, weights and output.
func.func @conv1d_float_type_mismatch(
    %input:   memref<16x16xf32>,
    %weights: memref<8x16x3xf32>,
    %output:  memref<8x14xf16>
) {
  // expected-error @+1 {{float mode requires input, weights, and output to share the same float element type}}
  snn.conv1d ins(%input, %weights) out(%output)
      : memref<16x16xf32>, memref<8x16x3xf32> -> memref<8x14xf16>
  return
}

// -----
// snn.sumpool2d output height must follow (H + 2p - Kh)/s + 1.
func.func @sumpool2d_bad_output_height(
    %input:  memref<16x16x16xf32>,
    %output: memref<16x9x8xf32>
) {
  // expected-error @+1 {{output height (9) does not match (H+2p-Kh)/s+1 (8)}}
  snn.sumpool2d ins(%input) out(%output)
      {kernel = array<i64: 2, 2>, stride = array<i64: 2, 2>}
      : memref<16x16x16xf32> -> memref<16x9x8xf32>
  return
}

// -----
// snn.sumpool2d preserves the channel count.
func.func @sumpool2d_channel_mismatch(
    %input:  memref<16x16x16xf32>,
    %output: memref<8x8x8xf32>
) {
  // expected-error @+1 {{pooling preserves channels: input (16) and output (8) channel counts must match}}
  snn.sumpool2d ins(%input) out(%output)
      {kernel = array<i64: 2, 2>, stride = array<i64: 2, 2>}
      : memref<16x16x16xf32> -> memref<8x8x8xf32>
  return
}

// -----
// snn.sumpool2d in float mode is type-uniform across input and output.
func.func @sumpool2d_type_mismatch(
    %input:  memref<16x16x16xf32>,
    %output: memref<16x8x8xf16>
) {
  // expected-error @+1 {{float mode requires input and output to share the same float element type}}
  snn.sumpool2d ins(%input) out(%output)
      {kernel = array<i64: 2, 2>, stride = array<i64: 2, 2>}
      : memref<16x16x16xf32> -> memref<16x8x8xf16>
  return
}

// -----
// snn.avgpool2d output height must follow (H + 2p - Kh)/s + 1.
func.func @avgpool2d_bad_output_height(
    %input:  memref<16x16x16xf32>,
    %output: memref<16x9x8xf32>
) {
  // expected-error @+1 {{output height (9) does not match (H+2p-Kh)/s+1 (8)}}
  snn.avgpool2d ins(%input) out(%output)
      {kernel = array<i64: 2, 2>, stride = array<i64: 2, 2>}
      : memref<16x16x16xf32> -> memref<16x9x8xf32>
  return
}

// -----
// snn.avgpool2d preserves the channel count.
func.func @avgpool2d_channel_mismatch(
    %input:  memref<16x16x16xf32>,
    %output: memref<8x8x8xf32>
) {
  // expected-error @+1 {{pooling preserves channels: input (16) and output (8) channel counts must match}}
  snn.avgpool2d ins(%input) out(%output)
      {kernel = array<i64: 2, 2>, stride = array<i64: 2, 2>}
      : memref<16x16x16xf32> -> memref<8x8x8xf32>
  return
}

// -----
// snn.avgpool2d quantized contract is i8 -> i8 (a truncating integer mean).
func.func @avgpool2d_bad_quant_output(
    %input:  memref<16x16x16xi8>,
    %output: memref<16x8x8xi32>
) {
  // expected-error @+1 {{quantized mode is i8 -> i8 (truncating integer mean)}}
  snn.avgpool2d ins(%input) out(%output)
      {kernel = array<i64: 2, 2>, stride = array<i64: 2, 2>}
      : memref<16x16x16xi8> -> memref<16x8x8xi32>
  return
}
