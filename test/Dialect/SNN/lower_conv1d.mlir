// Copyright 2026 N Vision Systems And Technologies SL
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Test: snn.conv1d lowers to linalg.conv_1d_ncw_fcw (float) and, with no rank-3
// quantized named conv available, to linalg.conv_2d_nchw_fchw_q via a 2-D embed.
// RUN: %snn-opt --convert-snn-to-linalg %s | %FileCheck %s

// No padding, unit stride, no bias: the rank-2 activations are bracketed with a
// unit batch dimension (memref.expand_shape) so the named rank-3 conv applies;
// the output is zeroed first because the conv accumulates.
// CHECK-LABEL: func.func @conv_plain
// CHECK-NOT:     memref.subview
// CHECK:         memref.expand_shape %arg0 {{\[\[}}0, 1], [2]] {{.*}} into memref<1x16x16xf32>
// CHECK:         memref.expand_shape %arg2 {{\[\[}}0, 1], [2]] {{.*}} into memref<1x8x14xf32>
// CHECK:         linalg.fill
// CHECK:         linalg.conv_1d_ncw_fcw {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
// CHECK-NOT:     snn.conv1d
func.func @conv_plain(
    %in:  memref<16x16xf32>,
    %w:   memref<8x16x3xf32>,
    %out: memref<8x14xf32>
) {
  snn.conv1d ins(%in, %w) out(%out)
      : memref<16x16xf32>, memref<8x16x3xf32> -> memref<8x14xf32>
  return
}

// Padding materialized: a zero-filled padded buffer with the input copied into
// its interior subview, then the conv over the padded buffer. Stride 2.
// CHECK-LABEL: func.func @conv_pad_stride
// CHECK:         %[[PAD:.*]] = memref.alloca() : memref<2x36xf32>
// CHECK:         linalg.fill ins({{.*}}) outs(%[[PAD]]
// CHECK:         memref.subview %[[PAD]][0, 1] [2, 34] [1, 1]
// CHECK:         linalg.copy ins(%arg0
// CHECK:         linalg.conv_1d_ncw_fcw {dilations = dense<1> : tensor<1xi64>, strides = dense<2> : tensor<1xi64>}
// CHECK-NOT:     snn.conv1d
func.func @conv_pad_stride(
    %in:  memref<2x34xf32>,
    %w:   memref<16x2x5xf32>,
    %out: memref<16x16xf32>
) {
  snn.conv1d ins(%in, %w) out(%out)
      {stride = 2 : i64, padding = 1 : i64}
      : memref<2x34xf32>, memref<16x2x5xf32> -> memref<16x16xf32>
  return
}

// Bias broadcast: a per-output-channel value added over every spatial position
// by a trailing linalg.generic mapping the channel index onto (o, l).
// CHECK-LABEL: func.func @conv_bias
// CHECK:         linalg.conv_1d_ncw_fcw
// CHECK:         linalg.generic
// CHECK:         arith.addf
// CHECK-NOT:     snn.conv1d
func.func @conv_bias(
    %in:  memref<16x16xf32>,
    %w:   memref<8x16x3xf32>,
    %b:   memref<8xf32>,
    %out: memref<8x14xf32>
) {
  snn.conv1d ins(%in, %w) bias(%b : memref<8xf32>) out(%out)
      : memref<16x16xf32>, memref<8x16x3xf32> -> memref<8x14xf32>
  return
}

// Quantized: no rank-3 quantized named conv exists, so the 1-D convolution is
// embedded in a 2-D one with a unit width axis — [C, L] activations become
// [1, C, Lp, 1], [O, C, K] weights become [O, C, K, 1] — and
// conv_2d_nchw_fchw_q applies. Both zero-points are 0 (symmetric); the padded
// buffer is zero-filled with an i8 zero; bias is i32, added with arith.addi.
// CHECK-LABEL: func.func @conv1d_quant
// CHECK:         %[[PAD:.*]] = memref.alloca() : memref<2x36xi8>
// CHECK:         linalg.fill ins({{.*}}) outs(%[[PAD]]
// CHECK:         memref.expand_shape {{.*}} {{\[\[}}0, 1], [2, 3]] {{.*}} into memref<1x2x36x1xi8>
// CHECK:         memref.expand_shape %arg1 {{\[\[}}0], [1], [2, 3]] {{.*}} into memref<16x2x5x1xi8>
// CHECK:         memref.expand_shape %arg3 {{\[\[}}0, 1], [2, 3]] {{.*}} into memref<1x16x16x1xi32>
// CHECK:         linalg.conv_2d_nchw_fchw_q {{.*}} ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : memref<1x2x36x1xi8>, memref<16x2x5x1xi8>, i32, i32)
// CHECK:         linalg.generic
// CHECK:         arith.addi
// CHECK-NOT:     snn.conv1d
func.func @conv1d_quant(
    %in:  memref<2x34xi8>,
    %w:   memref<16x2x5xi8>,
    %b:   memref<16xi32>,
    %out: memref<16x16xi32>
) {
  snn.conv1d ins(%in, %w) bias(%b : memref<16xi32>) out(%out)
      {stride = 2 : i64, padding = 1 : i64, w_scale = 5 : i64}
      : memref<2x34xi8>, memref<16x2x5xi8> -> memref<16x16xi32>
  return
}
