// Copyright 2026 Sensing & Control Systems, S.L.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Test: verify each SNN dialect op parses and round-trips correctly in isolation.
// RUN: %snn-opt %s | %FileCheck %s

// ── snn.linear ──────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @linear_float
// CHECK: snn.linear
func.func @linear_float(
    %input:   memref<32xf32>,
    %weights: memref<64x32xf32>,
    %out:     memref<64xf32>
) {
  snn.linear ins(%input, %weights) out(%out)
      : memref<32xf32>, memref<64x32xf32> -> memref<64xf32>
  return
}

// CHECK-LABEL: func.func @linear_quantized
// CHECK: snn.linear
func.func @linear_quantized(
    %input:   memref<32xi8>,
    %weights: memref<64x32xi8>,
    %out:     memref<64xi32>
) {
  snn.linear ins(%input, %weights) out(%out) {w_scale = 7 : i64}
      : memref<32xi8>, memref<64x32xi8> -> memref<64xi32>
  return
}

// CHECK-LABEL: func.func @linear_float_bias
// CHECK: snn.linear
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

// CHECK-LABEL: func.func @linear_quantized_bias
// CHECK: snn.linear
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

// ── snn.rescale ─────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @rescale
// CHECK: snn.rescale
func.func @rescale(
    %input:  memref<64xi32>,
    %output: memref<64xi32>
) {
  snn.rescale ins(%input) out(%output) {w_scale = 7 : i64, d_scale = 12 : i64}
      : memref<64xi32> -> memref<64xi32>
  return
}

// ── snn.cubalif ─────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @cubalif_float
// CHECK: snn.cubalif
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

// CHECK-LABEL: func.func @cubalif_quantized
// CHECK: snn.cubalif
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

// ── snn.cubali ──────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @cubali_float
// CHECK: snn.cubali
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

// CHECK-LABEL: func.func @cubali_quantized
// CHECK: snn.cubali
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

// ── snn.lif ─────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @lif_float
// CHECK: snn.lif
func.func @lif_float(
    %input:   memref<64xf32>,
    %voltage: memref<64xf32>,
    %output:  memref<64xf32>
) {
  snn.lif ins(%input) state(%voltage) out(%output)
      {decay_float = 9.500000e-01 : f64,
       threshold_float = 1.000000e+00 : f64}
      : memref<64xf32>, memref<64xf32> -> memref<64xf32>
  return
}

// CHECK-LABEL: func.func @lif_quantized
// CHECK: snn.lif
func.func @lif_quantized(
    %input:   memref<64xi32>,
    %voltage: memref<64xi32>,
    %output:  memref<64xi8>
) {
  snn.lif ins(%input) state(%voltage) out(%output)
      {d_scale = 12 : i64,
       decay_int = 3891 : i64,
       threshold_int = 4096 : i64}
      : memref<64xi32>, memref<64xi32> -> memref<64xi8>
  return
}

// ── snn.li ──────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @li_float
// CHECK: snn.li
func.func @li_float(
    %input:   memref<64xf32>,
    %voltage: memref<64xf32>,
    %output:  memref<64xf32>
) {
  snn.li ins(%input) state(%voltage) out(%output)
      {decay_float = 9.500000e-01 : f64}
      : memref<64xf32>, memref<64xf32> -> memref<64xf32>
  return
}

// CHECK-LABEL: func.func @li_quantized
// CHECK: snn.li
func.func @li_quantized(
    %input:   memref<64xi32>,
    %voltage: memref<64xi32>,
    %output:  memref<64xi32>
) {
  snn.li ins(%input) state(%voltage) out(%output)
      {d_scale = 12 : i64, decay_int = 3891 : i64}
      : memref<64xi32>, memref<64xi32> -> memref<64xi32>
  return
}
