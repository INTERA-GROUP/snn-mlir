# Copyright 2026 Sensing & Control Systems, S.L.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
add_mlir_conversion_library(MLIRSNNToLinalg
  SNNToLinalg.cpp

  DEPENDS
  MLIRSNNOpsIncGen
  MLIRSNNConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRSNN
  MLIRIR
  MLIRPass
  MLIRLinalgDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRTransforms
)
