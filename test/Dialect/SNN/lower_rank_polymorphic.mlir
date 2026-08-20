// Copyright 2026 N Vision Systems And Technologies SL
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Test: the neuron ops and snn.rescale are rank-polymorphic.
//
// These ops are elementwise and shape-preserving, so their iteration space is
// simply the operand shape — one identity map per operand, one parallel
// iterator per dimension, at whatever rank the operands carry. A dense layer
// hands them a vector; a convolutional layer hands them a (C, H, W) feature map
// and expects the same neuron, not a reshape at every layer boundary.
//
// What is pinned here is the ITERATION SPACE only. The arithmetic inside each
// generic is rank-independent and is already pinned per op by lower_lif.mlir,
// lower_cubalif.mlir, lower_li.mlir, lower_cubali.mlir and lower_rescale.mlir;
// re-checking it here would duplicate those goldens without adding coverage.
//
// snn.linear is deliberately absent: it stays strictly 1-D. Crossing between a
// feature map and a vector is memref.collapse_shape's job, spelled out in the
// IR rather than implied by an op that quietly accepts either.
// RUN: %snn-opt --convert-snn-to-linalg %s | %FileCheck %s

// CHECK-DAG: #[[$MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[$MAP4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// ── snn.rescale ─────────────────────────────────────────────────────────────

// A conv accumulator requantized on its way into a neuron, still shaped as a
// feature map.
// CHECK-LABEL: func.func @rescale_rank3
// CHECK:         linalg.generic {indexing_maps = [#[[$MAP3]], #[[$MAP3]]], iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK:         arith.shli
// CHECK-NOT:     snn.rescale
func.func @rescale_rank3(
    %input:  memref<16x16x16xi32>,
    %output: memref<16x16x16xi32>
) {
  snn.rescale ins(%input) out(%output) {w_scale = 7 : i64, d_scale = 12 : i64}
      : memref<16x16x16xi32> -> memref<16x16x16xi32>
  return
}

// ── snn.lif ─────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @lif_rank3_float
// CHECK:         linalg.generic {indexing_maps = [#[[$MAP3]], #[[$MAP3]], #[[$MAP3]]], iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-NOT:     snn.lif
func.func @lif_rank3_float(
    %input:   memref<16x16x16xf32>,
    %voltage: memref<16x16x16xf32>,
    %output:  memref<16x16x16xf32>
) {
  snn.lif ins(%input) state(%voltage) out(%output)
      {decay_float = 9.500000e-01 : f64,
       threshold_float = 1.000000e+00 : f64}
      : memref<16x16x16xf32>, memref<16x16x16xf32> -> memref<16x16x16xf32>
  return
}

// The quantized shape a conv layer's neuron actually takes: i32 state in, i8
// spikes out, both carrying the feature map's geometry.
// CHECK-LABEL: func.func @lif_rank3_quantized
// CHECK:         linalg.generic {indexing_maps = [#[[$MAP3]], #[[$MAP3]], #[[$MAP3]]], iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-NOT:     snn.lif
func.func @lif_rank3_quantized(
    %input:   memref<16x16x16xi32>,
    %voltage: memref<16x16x16xi32>,
    %output:  memref<16x16x16xi8>
) {
  snn.lif ins(%input) state(%voltage) out(%output)
      {d_scale = 12 : i64,
       decay_int = 4096 : i64,
       threshold_int = 4096 : i64}
      : memref<16x16x16xi32>, memref<16x16x16xi32> -> memref<16x16x16xi8>
  return
}

// ── snn.cubalif ─────────────────────────────────────────────────────────────

// Two state operands, so four identity maps rather than three.
// CHECK-LABEL: func.func @cubalif_rank3_quantized
// CHECK:         linalg.generic {indexing_maps = [#[[$MAP3]], #[[$MAP3]], #[[$MAP3]], #[[$MAP3]]], iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-NOT:     snn.cubalif
func.func @cubalif_rank3_quantized(
    %input:   memref<8x12x12xi32>,
    %current: memref<8x12x12xi32>,
    %voltage: memref<8x12x12xi32>,
    %output:  memref<8x12x12xi8>
) {
  snn.cubalif ins(%input) state(%current, %voltage) out(%output)
      {d_scale = 12 : i64, cur_decay_int = 3686 : i64,
       vol_decay_int = 4055 : i64, threshold_int = 4096 : i64}
      : memref<8x12x12xi32>, memref<8x12x12xi32>, memref<8x12x12xi32>
      -> memref<8x12x12xi8>
  return
}

// ── snn.li ──────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @li_rank3_float
// CHECK:         linalg.generic {indexing_maps = [#[[$MAP3]], #[[$MAP3]], #[[$MAP3]]], iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-NOT:     snn.li
func.func @li_rank3_float(
    %input:   memref<8x12x12xf32>,
    %voltage: memref<8x12x12xf32>,
    %output:  memref<8x12x12xf32>
) {
  snn.li ins(%input) state(%voltage) out(%output)
      {decay_float = 9.500000e-01 : f64}
      : memref<8x12x12xf32>, memref<8x12x12xf32> -> memref<8x12x12xf32>
  return
}

// ── snn.cubali ──────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @cubali_rank3_float
// CHECK:         linalg.generic {indexing_maps = [#[[$MAP3]], #[[$MAP3]], #[[$MAP3]], #[[$MAP3]]], iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-NOT:     snn.cubali
func.func @cubali_rank3_float(
    %input:   memref<8x12x12xf32>,
    %current: memref<8x12x12xf32>,
    %voltage: memref<8x12x12xf32>,
    %output:  memref<8x12x12xf32>
) {
  snn.cubali ins(%input) state(%current, %voltage) out(%output)
      {cur_decay_float = 9.000000e-01 : f64,
       vol_decay_float = 9.500000e-01 : f64}
      : memref<8x12x12xf32>, memref<8x12x12xf32>, memref<8x12x12xf32>
      -> memref<8x12x12xf32>
  return
}

// ── rank is derived, not assumed ────────────────────────────────────────────

// Rank 3 is what a (C, H, W) feature map happens to need; nothing in the
// lowering knows that number. A rank-4 operand must produce a rank-4 iteration
// space by the same rule.
// CHECK-LABEL: func.func @lif_rank4_quantized
// CHECK:         linalg.generic {indexing_maps = [#[[$MAP4]], #[[$MAP4]], #[[$MAP4]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
// CHECK-NOT:     snn.lif
func.func @lif_rank4_quantized(
    %input:   memref<2x16x16x16xi32>,
    %voltage: memref<2x16x16x16xi32>,
    %output:  memref<2x16x16x16xi8>
) {
  snn.lif ins(%input) state(%voltage) out(%output)
      {d_scale = 12 : i64,
       decay_int = 4096 : i64,
       threshold_int = 4096 : i64}
      : memref<2x16x16x16xi32>, memref<2x16x16x16xi32> -> memref<2x16x16x16xi8>
  return
}
