set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name SNNConversion)
add_public_tablegen_target(MLIRSNNConversionPassIncGen)

# Auto-generated pass documentation (build with: ninja mlir-doc)
add_mlir_doc(Passes SNNConversionPasses Conversion/ -gen-pass-doc)
