// Copyright 2026 N Vision Systems And Technologies SL
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Test: snn.cubali lowers to two-decay dynamics with continuous voltage output.
// No threshold comparison or spike selection should appear.
// RUN: %snn-opt --convert-snn-to-linalg %s | %FileCheck %s

// Float path: mulf/addf for both decay steps; no cmpf or spike select.
// CHECK-LABEL: func.func @cubali_float
// CHECK:         linalg.generic
// CHECK:         arith.mulf
// CHECK:         arith.addf
// CHECK-NOT:     arith.cmpf
// CHECK-NOT:     snn.cubali
func.func @cubali_float(
    %input:   memref<64xf32>,
    %current: memref<64xf32>,
    %voltage: memref<64xf32>,
    %output:  memref<64xf32>
) {
  snn.cubali ins(%input) state(%current, %voltage) out(%output)
      {cur_decay_float = 9.000000e-01 : f64,
       vol_decay_float = 9.500000e-01 : f64}
      : memref<64xf32>, memref<64xf32>, memref<64xf32> -> memref<64xf32>
  return
}

// Quantized path: muli/shrsi/addi for both decay steps; no cmpi or spike select.
// CHECK-LABEL: func.func @cubali_quantized
// CHECK:         linalg.generic
// CHECK:         arith.muli
// CHECK:         arith.shrsi
// CHECK:         arith.addi
// CHECK-NOT:     arith.cmpi
// CHECK-NOT:     snn.cubali
func.func @cubali_quantized(
    %input:   memref<64xi32>,
    %current: memref<64xi32>,
    %voltage: memref<64xi32>,
    %output:  memref<64xi32>
) {
  snn.cubali ins(%input) state(%current, %voltage) out(%output)
      {d_scale = 12 : i64, cur_decay_int = 3686 : i64,
       vol_decay_int = 4055 : i64}
      : memref<64xi32>, memref<64xi32>, memref<64xi32> -> memref<64xi32>
  return
}
