add_llvm_executable(snn-opt snn-opt.cpp)
add_dependencies(snn-opt MLIRSNNOpsIncGen)
target_link_libraries(snn-opt PRIVATE
  MLIRSNN
  MLIRSNNToLinalg
  MLIRMlirOptMain
)