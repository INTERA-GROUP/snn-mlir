set(LLVM_TARGET_DEFINITIONS SNNOps.td)

mlir_tablegen(SNNOps.h.inc       -gen-op-decls)
mlir_tablegen(SNNOps.cpp.inc     -gen-op-defs)
mlir_tablegen(SNNDialect.h.inc   -gen-dialect-decls)
mlir_tablegen(SNNDialect.cpp.inc -gen-dialect-defs)

add_public_tablegen_target(MLIRSNNOpsIncGen)

# Auto-generated dialect + op reference documentation (build with: ninja mlir-doc)
add_mlir_doc(SNNOps SNNDialect Dialects/ -gen-dialect-doc)

# Conversion-pass TableGen declarations (convert-snn-to-linalg)
add_subdirectory(Conversion)
