// Test: snn.lif lowers to single-state dynamics with threshold and spike output.
// Threshold comparison uses strict greater-than (cmpf ogt / cmpi sgt), matching
// snntorch's fire() semantics (S=1 iff U > U_thr).
// RUN: %snn-opt --convert-snn-to-linalg %s | %FileCheck %s

// Float path: mulf/addf for decay; cmpf ogt for threshold; select for voltage reset and spike.
// CHECK-LABEL: func.func @lif_float
// CHECK:         linalg.generic
// CHECK:         arith.mulf
// CHECK:         arith.addf
// CHECK:         arith.cmpf ogt
// CHECK:         arith.select
// CHECK-NOT:     snn.lif
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

// Quantized path: muli/shrsi/addi for Q12 decay; cmpi sgt for threshold; i8 spike output.
// CHECK-LABEL: func.func @lif_quantized
// CHECK:         linalg.generic
// CHECK:         arith.muli
// CHECK:         arith.shrsi
// CHECK:         arith.addi
// CHECK:         arith.cmpi sgt
// CHECK:         arith.select
// CHECK-NOT:     snn.lif
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
