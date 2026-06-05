# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
cmake_minimum_required(VERSION 3.20.0)
project(snn-dialect LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

# When built standalone (not as part of a larger project)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  find_package(MLIR 22.1 REQUIRED CONFIG)

  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)

  if(NOT LLVM_ENABLE_RTTI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
  endif()

  include_directories(${LLVM_INCLUDE_DIRS})
  include_directories(${MLIR_INCLUDE_DIRS})

  # Locate lit and FileCheck for ninja check-snn.
  # Users can also pass -DLLVM_EXTERNAL_LIT=<path> explicitly.
  if(NOT LLVM_EXTERNAL_LIT)
    find_program(LLVM_EXTERNAL_LIT
      NAMES llvm-lit lit
      HINTS ${LLVM_TOOLS_BINARY_DIR}
      DOC "llvm-lit test runner"
    )
  endif()
  find_program(FILECHECK_EXE
    NAMES FileCheck
    HINTS ${LLVM_TOOLS_BINARY_DIR}
    DOC "FileCheck utility"
  )
  set(SNN_OPT_ABS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/snn-opt")
  if(LLVM_EXTERNAL_LIT AND FILECHECK_EXE)
    message(STATUS "Found lit: ${LLVM_EXTERNAL_LIT} — ninja check-snn available")
  else()
    message(STATUS "llvm-lit or FileCheck not found — ninja check-snn will not be available")
  endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)