// Copyright 2026 N Vision Systems And Technologies SL
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "SNN/SNNDialect.h"
#include "SNN/Conversion/SNNToLinalg.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"


int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();
  registry.insert<snn::SNNDialect>();
  snn::registerConvertSNNToLinalgPass();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "SNN dialect optimizer\n", registry));
}
