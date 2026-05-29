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
