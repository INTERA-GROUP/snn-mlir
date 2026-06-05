# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
add_llvm_executable(snn-opt snn-opt.cpp)
add_dependencies(snn-opt MLIRSNNOpsIncGen)
target_link_libraries(snn-opt PRIVATE
  MLIRSNN
  MLIRSNNToLinalg
  MLIRMlirOptMain
)