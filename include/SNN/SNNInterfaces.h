// Copyright 2026 N Vision Systems And Technologies SL
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===- SNNInterfaces.h - SNN op interface declarations ---------*- C++ -*-===//
#ifndef SNN_INTERFACES_H
#define SNN_INTERFACES_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"

// Pull in the auto-generated interface declarations. Must be included before the
// op declarations (SNNOps.h.inc), which reference these interface base classes.
#include "SNN/SNNInterfaces.h.inc"

#endif // SNN_INTERFACES_H
