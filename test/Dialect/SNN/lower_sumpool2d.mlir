// Copyright 2026 N Vision Systems And Technologies SL
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Test: snn.sumpool2d lowers correctly to linalg.pooling_nchw_sum (float).
// RUN: %snn-opt --convert-snn-to-linalg %s | %FileCheck %s

// No padding, 2x2 window, stride 2 (the nmnistcnn case): the rank-3 activations
// are bracketed with a unit batch dimension (memref.expand_shape) so the named
// pooling op applies; the output is zeroed first because pooling accumulates,
// and a throwaway window alloca supplies the [Kh, Kw] shape the op reduces over.
// CHECK-LABEL: func.func @pool_plain
// CHECK-NOT:     memref.subview
// CHECK:         memref.expand_shape %arg0 {{\[\[}}0, 1], [2], [3]] {{.*}} into memref<1x16x16x16xf32>
// CHECK:         memref.expand_shape %arg1 {{\[\[}}0, 1], [2], [3]] {{.*}} into memref<1x16x8x8xf32>
// CHECK:         linalg.fill
// CHECK:         %[[WIN:.*]] = memref.alloca() : memref<2x2xf32>
// CHECK:         linalg.pooling_nchw_sum {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
// CHECK-NOT:     snn.sumpool2d
func.func @pool_plain(
    %in:  memref<16x16x16xf32>,
    %out: memref<16x8x8xf32>
) {
  snn.sumpool2d ins(%in) out(%out)
      {kernel = array<i64: 2, 2>, stride = array<i64: 2, 2>}
      : memref<16x16x16xf32> -> memref<16x8x8xf32>
  return
}

// Padding materialized: a zero-filled padded buffer with the input copied into
// its interior subview, then the pooling over the padded buffer.
// CHECK-LABEL: func.func @pool_pad
// CHECK:         %[[PAD:.*]] = memref.alloca() : memref<8x10x10xf32>
// CHECK:         linalg.fill ins({{.*}}) outs(%[[PAD]]
// CHECK:         memref.subview %[[PAD]][0, 1, 1] [8, 8, 8] [1, 1, 1]
// CHECK:         linalg.copy ins(%arg0
// CHECK:         linalg.pooling_nchw_sum {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
// CHECK-NOT:     snn.sumpool2d
func.func @pool_pad(
    %in:  memref<8x8x8xf32>,
    %out: memref<8x5x5xf32>
) {
  snn.sumpool2d ins(%in) out(%out)
      {kernel = array<i64: 2, 2>, stride = array<i64: 2, 2>, padding = array<i64: 1, 1>}
      : memref<8x8x8xf32> -> memref<8x5x5xf32>
  return
}

// Quantized: sum pooling is scale-preserving, so the element contract is i8 ->
// i8. Same named pooling op, i8 fills and window.
// CHECK-LABEL: func.func @pool_quant
// CHECK:         linalg.fill
// CHECK:         linalg.pooling_nchw_sum {{.*}} ins(%{{.*}}, %{{.*}} : memref<1x8x8x8xi8>, memref<2x2xi8>) outs(%{{.*}} : memref<1x8x4x4xi8>)
// CHECK-NOT:     snn.sumpool2d
func.func @pool_quant(
    %in:  memref<8x8x8xi8>,
    %out: memref<8x4x4xi8>
) {
  snn.sumpool2d ins(%in) out(%out)
      {kernel = array<i64: 2, 2>, stride = array<i64: 2, 2>}
      : memref<8x8x8xi8> -> memref<8x4x4xi8>
  return
}
