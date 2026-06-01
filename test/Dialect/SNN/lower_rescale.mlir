// Copyright 2026 Sensing & Control Systems, S.L.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Test: snn.rescale lowers to a linalg.generic with the correct shift direction.
// RUN: %snn-opt --convert-snn-to-linalg %s | %FileCheck %s

// Positive shift (d_scale > w_scale): left-shift to upscale the accumulator.
// CHECK-LABEL: func.func @rescale_shift_up
// CHECK:         linalg.generic
// CHECK:         arith.shli
// CHECK-NOT:     snn.rescale
func.func @rescale_shift_up(
    %input:  memref<64xi32>,
    %output: memref<64xi32>
) {
  snn.rescale ins(%input) out(%output) {w_scale = 7 : i64, d_scale = 12 : i64}
      : memref<64xi32> -> memref<64xi32>
  return
}

// Negative shift (d_scale < w_scale): arithmetic right-shift to downscale.
// CHECK-LABEL: func.func @rescale_shift_down
// CHECK:         linalg.generic
// CHECK:         arith.shrsi
// CHECK-NOT:     snn.rescale
func.func @rescale_shift_down(
    %input:  memref<64xi32>,
    %output: memref<64xi32>
) {
  snn.rescale ins(%input) out(%output) {w_scale = 12 : i64, d_scale = 7 : i64}
      : memref<64xi32> -> memref<64xi32>
  return
}
