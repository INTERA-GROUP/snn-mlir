# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Op interfaces (SynapseOpInterface, NeuronOpInterface). Generated first: the op
# declarations below reference the interface base classes.
set(LLVM_TARGET_DEFINITIONS SNNInterfaces.td)
mlir_tablegen(SNNInterfaces.h.inc   -gen-op-interface-decls)
mlir_tablegen(SNNInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRSNNInterfacesIncGen)

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
