// Test: snn.li lowers to single-state leaky integration with continuous voltage output.
// No threshold comparison or spike selection should appear.
// RUN: %snn-opt --convert-snn-to-linalg %s | %FileCheck %s

// Float path: mulf/addf for decay; continuous voltage output; no cmpf.
// CHECK-LABEL: func.func @li_float
// CHECK:         linalg.generic
// CHECK:         arith.mulf
// CHECK:         arith.addf
// CHECK-NOT:     arith.cmpf
// CHECK-NOT:     snn.li
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

// Quantized path: muli/shrsi/addi for Q12 decay; no cmpi.
// CHECK-LABEL: func.func @li_quantized
// CHECK:         linalg.generic
// CHECK:         arith.muli
// CHECK:         arith.shrsi
// CHECK:         arith.addi
// CHECK-NOT:     arith.cmpi
// CHECK-NOT:     snn.li
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
