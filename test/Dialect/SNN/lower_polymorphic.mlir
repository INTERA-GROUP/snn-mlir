// Test: the float lowering is type-polymorphic over FloatType width.
// The neuron lowerings derive their element type from the operand memref, so
// f64 / f16 networks lower to f64 / f16 arithmetic — not hardcoded f32.
// RUN: %snn-opt --convert-snn-to-linalg %s | %FileCheck %s

// f64 CubaLIF: every materialised constant must be f64, and no f32 may appear.
// CHECK-LABEL: func.func @cubalif_f64
// CHECK:         linalg.generic
// CHECK:         arith.constant {{.*}} : f64
// CHECK:         arith.mulf
// CHECK:         arith.cmpf ogt
// CHECK-NOT:     : f32
// CHECK-NOT:     snn.cubalif
func.func @cubalif_f64(
    %input:   memref<64xf64>,
    %current: memref<64xf64>,
    %voltage: memref<64xf64>,
    %output:  memref<64xf64>
) {
  snn.cubalif ins(%input) state(%current, %voltage) out(%output)
      {cur_decay_float = 9.000000e-01 : f64,
       vol_decay_float = 9.500000e-01 : f64,
       threshold_float = 1.000000e+00 : f64}
      : memref<64xf64>, memref<64xf64>, memref<64xf64> -> memref<64xf64>
  return
}

// -----

// f16 LIF: constants and arithmetic must be f16.
// CHECK-LABEL: func.func @lif_f16
// CHECK:         linalg.generic
// CHECK:         arith.constant {{.*}} : f16
// CHECK:         arith.mulf
// CHECK:         arith.cmpf ogt
// CHECK-NOT:     : f32
// CHECK-NOT:     snn.lif
func.func @lif_f16(
    %input:   memref<64xf16>,
    %voltage: memref<64xf16>,
    %output:  memref<64xf16>
) {
  snn.lif ins(%input) state(%voltage) out(%output)
      {decay_float = 9.500000e-01 : f64,
       threshold_float = 1.000000e+00 : f64}
      : memref<64xf16>, memref<64xf16> -> memref<64xf16>
  return
}
