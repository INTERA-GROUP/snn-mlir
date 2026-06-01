# Copyright 2026 Sensing & Control Systems, S.L.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
add_mlir_dialect_library(MLIRSNN
  SNNDialect.cpp
  SNNOps.cpp

  DEPENDS
  MLIRSNNOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
)