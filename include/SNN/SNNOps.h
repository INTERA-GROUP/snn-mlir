// Copyright 2026 N Vision Systems And Technologies SL
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===------------------------------ SNN op declarations --------------------------*- C++ -*-===//
#ifndef SNN_OPS_H
#define SNN_OPS_H

#include "SNN/SNNDialect.h"
#include "SNN/SNNInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Pull in the auto-generated op declarations. SNNInterfaces.h (above) must
// precede this: the op classes derive from the generated interface bases.
#define GET_OP_CLASSES
#include "SNN/SNNOps.h.inc"

#endif // SNN_OPS_H
