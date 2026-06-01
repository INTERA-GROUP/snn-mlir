// Copyright 2026 Sensing & Control Systems, S.L.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Test: snn.linear lowers correctly to linalg/arith ops.
// RUN: %snn-opt --convert-snn-to-linalg %s | %FileCheck %s

// Float path: linalg.fill zeros the output, linalg.matvec computes weights @ input.
// CHECK-LABEL: func.func @linear_float
// CHECK:         linalg.fill
// CHECK:         linalg.matvec
// CHECK-NOT:     snn.linear
func.func @linear_float(
    %input:   memref<32xf32>,
    %weights: memref<64x32xf32>,
    %out:     memref<64xf32>
) {
  snn.linear ins(%input, %weights) out(%out)
      : memref<32xf32>, memref<64x32xf32> -> memref<64xf32>
  return
}

// Quantized path: i8 inputs sign-extended to i32, multiplied and accumulated.
// CHECK-LABEL: func.func @linear_quantized
// CHECK:         linalg.fill
// CHECK:         linalg.generic
// CHECK:         arith.extsi
// CHECK:         arith.muli
// CHECK:         arith.addi
// CHECK-NOT:     snn.linear
func.func @linear_quantized(
    %input:   memref<32xi8>,
    %weights: memref<64x32xi8>,
    %out:     memref<64xi32>
) {
  snn.linear ins(%input, %weights) out(%out) {w_scale = 7 : i64}
      : memref<32xi8>, memref<64x32xi8> -> memref<64xi32>
  return
}

// Float with bias: matvec followed by a second linalg.generic that adds bias elementwise.
// CHECK-LABEL: func.func @linear_float_bias
// CHECK:         linalg.matvec
// CHECK:         linalg.generic
// CHECK:         arith.addf
// CHECK-NOT:     snn.linear
func.func @linear_float_bias(
    %input:   memref<32xf32>,
    %weights: memref<64x32xf32>,
    %bias:    memref<64xf32>,
    %out:     memref<64xf32>
) {
  snn.linear ins(%input, %weights) bias(%bias : memref<64xf32>) out(%out)
      : memref<32xf32>, memref<64x32xf32> -> memref<64xf32>
  return
}

// Quantized with bias: generic matmul followed by a second generic that adds i32 bias.
// CHECK-LABEL: func.func @linear_quantized_bias
// CHECK:         linalg.generic
// CHECK:         arith.extsi
// CHECK:         linalg.generic
// CHECK:         arith.addi
// CHECK-NOT:     snn.linear
func.func @linear_quantized_bias(
    %input:   memref<32xi8>,
    %weights: memref<64x32xi8>,
    %bias:    memref<64xi32>,
    %out:     memref<64xi32>
) {
  snn.linear ins(%input, %weights) bias(%bias : memref<64xi32>) out(%out)
      {w_scale = 7 : i64}
      : memref<32xi8>, memref<64x32xi8> -> memref<64xi32>
  return
}
