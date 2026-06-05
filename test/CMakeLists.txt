# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
if(LLVM_EXTERNAL_LIT AND FILECHECK_EXE)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py"
  )

  add_lit_testsuite(check-snn "Running SNN dialect regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS snn-opt
  )
  set_target_properties(check-snn PROPERTIES FOLDER "Tests")
endif()

add_subdirectory(Dialect)
