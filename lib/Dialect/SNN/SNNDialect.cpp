// Copyright 2026 N Vision Systems And Technologies SL
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===- SNNDialect.cpp - SNN dialect implementation -------------*- C++ -*-===//

#include "SNN/SNNDialect.h"
#include "SNN/SNNOps.h"

// Pull in the auto-generated dialect method bodies
#include "SNN/SNNDialect.cpp.inc"

// This is called once when the dialect is loaded into an MLIRContext.
// It registers all the ops defined in SNNOps.td.
void snn::SNNDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "SNN/SNNOps.cpp.inc"
  >();
}
