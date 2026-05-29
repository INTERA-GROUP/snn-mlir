// SPDX-License-Identifier: Apache-2.0
//===- Passes.td - SNN conversion passes -------------------*- tablegen -*-===//

#ifndef SNN_CONVERSION_PASSES_TD
#define SNN_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertSNNToLinalg : Pass<"convert-snn-to-linalg", "mlir::ModuleOp"> {
  let summary = "Convert SNN dialect ops to Linalg/Arith for CPU execution";
  let description = [{
    Lowers the SNN dialect (`snn.linear`, `snn.rescale`, `snn.cubalif`,
    `snn.lif`, `snn.li`, `snn.cubali`) to standard `linalg`/`arith`/`memref`
    operations that any MLIR CPU backend can consume. Dispatches on operand
    element types: float operands lower to floating-point arithmetic, integer
    operands lower to Q12 fixed-point arithmetic.
  }];
  let constructor = "snn::createConvertSNNToLinalgPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
  ];
}

#endif // SNN_CONVERSION_PASSES_TD
