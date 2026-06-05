# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name SNNConversion)
add_public_tablegen_target(MLIRSNNConversionPassIncGen)

# Auto-generated pass documentation (build with: ninja mlir-doc)
add_mlir_doc(Passes SNNConversionPasses Conversion/ -gen-pass-doc)
