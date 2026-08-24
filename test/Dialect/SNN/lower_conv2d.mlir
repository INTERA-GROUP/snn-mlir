// Copyright 2026 N Vision Systems And Technologies SL
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Test: snn.conv2d lowers correctly to linalg.conv_2d_nchw_fchw (float).
// RUN: %snn-opt --convert-snn-to-linalg %s | %FileCheck %s

// No padding, unit stride, no bias: the rank-3 activations are bracketed with a
// unit batch dimension (memref.expand_shape) so the named rank-4 conv applies;
// the output is zeroed first because the conv accumulates.
// CHECK-LABEL: func.func @conv_plain
// CHECK-NOT:     memref.subview
// CHECK:         memref.expand_shape %arg0 {{\[\[}}0, 1], [2], [3]] {{.*}} into memref<1x16x16x16xf32>
// CHECK:         memref.expand_shape %arg2 {{\[\[}}0, 1], [2], [3]] {{.*}} into memref<1x8x14x14xf32>
// CHECK:         linalg.fill
// CHECK:         linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
// CHECK-NOT:     snn.conv2d
func.func @conv_plain(
    %in:  memref<16x16x16xf32>,
    %w:   memref<8x16x3x3xf32>,
    %out: memref<8x14x14xf32>
) {
  snn.conv2d ins(%in, %w) out(%out)
      : memref<16x16x16xf32>, memref<8x16x3x3xf32> -> memref<8x14x14xf32>
  return
}

// Padding materialized: a zero-filled padded buffer with the input copied into
// its interior subview, then the conv over the padded buffer. Stride 2.
// CHECK-LABEL: func.func @conv_pad_stride
// CHECK:         %[[PAD:.*]] = memref.alloca() : memref<2x36x36xf32>
// CHECK:         linalg.fill ins({{.*}}) outs(%[[PAD]]
// CHECK:         memref.subview %[[PAD]][0, 1, 1] [2, 34, 34] [1, 1, 1]
// CHECK:         memref.copy %arg0
// CHECK:         linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
// CHECK-NOT:     snn.conv2d
func.func @conv_pad_stride(
    %in:  memref<2x34x34xf32>,
    %w:   memref<16x2x5x5xf32>,
    %out: memref<16x16x16xf32>
) {
  snn.conv2d ins(%in, %w) out(%out)
      {stride = array<i64: 2, 2>, padding = array<i64: 1, 1>}
      : memref<2x34x34xf32>, memref<16x2x5x5xf32> -> memref<16x16x16xf32>
  return
}

// Bias broadcast: a per-output-channel value added over every spatial position
// by a trailing linalg.generic mapping the channel index onto (o, h, w).
// CHECK-LABEL: func.func @conv_bias
// CHECK:         linalg.conv_2d_nchw_fchw
// CHECK:         linalg.generic
// CHECK:         arith.addf
// CHECK-NOT:     snn.conv2d
func.func @conv_bias(
    %in:  memref<16x16x16xf32>,
    %w:   memref<8x16x3x3xf32>,
    %b:   memref<8xf32>,
    %out: memref<8x14x14xf32>
) {
  snn.conv2d ins(%in, %w) bias(%b : memref<8xf32>) out(%out)
      : memref<16x16x16xf32>, memref<8x16x3x3xf32> -> memref<8x14x14xf32>
  return
}
