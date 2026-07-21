# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
add_mlir_dialect_library(MLIRSNN
  SNNDialect.cpp
  SNNOps.cpp

  DEPENDS
  MLIRSNNInterfacesIncGen
  MLIRSNNOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
)