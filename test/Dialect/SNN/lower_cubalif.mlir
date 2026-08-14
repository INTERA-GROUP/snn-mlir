// Copyright 2026 N Vision Systems And Technologies SL
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Test: snn.cubalif lowers to two-decay dynamics with threshold and spike output.
// RUN: %snn-opt --convert-snn-to-linalg %s | %FileCheck %s

// Float path: mulf/addf for current and voltage updates; cmpf ogt for threshold.
// CHECK-LABEL: func.func @cubalif_float
// CHECK:         linalg.generic
// CHECK:         arith.mulf
// CHECK:         arith.addf
// CHECK:         arith.cmpf ogt
// CHECK:         arith.select
// CHECK-NOT:     snn.cubalif
func.func @cubalif_float(
    %input:   memref<64xf32>,
    %current: memref<64xf32>,
    %voltage: memref<64xf32>,
    %output:  memref<64xf32>
) {
  snn.cubalif ins(%input) state(%current, %voltage) out(%output)
      {cur_decay_float = 9.000000e-01 : f64,
       vol_decay_float = 9.500000e-01 : f64,
       threshold_float = 1.000000e+00 : f64}
      : memref<64xf32>, memref<64xf32>, memref<64xf32> -> memref<64xf32>
  return
}

// Quantized path: each Q12 decay product is taken in i64 and truncated back, while state,
// threshold and spike stay i32/i8. Both decay stages are pinned — a partial revert must fail.
// CHECK-LABEL: func.func @cubalif_quantized
// CHECK:         linalg.generic
// CHECK:         arith.extsi %{{.*}} : i32 to i64
// CHECK:         arith.muli %{{.*}} : i64
// CHECK:         arith.shrsi %{{.*}} : i64
// CHECK:         arith.trunci %{{.*}} : i64 to i32
// CHECK:         arith.addi %{{.*}} : i32
// CHECK:         arith.extsi %{{.*}} : i32 to i64
// CHECK:         arith.muli %{{.*}} : i64
// CHECK:         arith.shrsi %{{.*}} : i64
// CHECK:         arith.trunci %{{.*}} : i64 to i32
// CHECK:         arith.addi %{{.*}} : i32
// CHECK:         arith.cmpi sgt
// CHECK:         arith.select
// CHECK-NOT:     snn.cubalif
func.func @cubalif_quantized(
    %input:   memref<64xi32>,
    %current: memref<64xi32>,
    %voltage: memref<64xi32>,
    %output:  memref<64xi8>
) {
  snn.cubalif ins(%input) state(%current, %voltage) out(%output)
      {d_scale = 12 : i64, cur_decay_int = 3686 : i64,
       vol_decay_int = 4055 : i64, threshold_int = 4096 : i64}
      : memref<64xi32>, memref<64xi32>, memref<64xi32> -> memref<64xi8>
  return
}
