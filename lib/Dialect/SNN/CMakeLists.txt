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