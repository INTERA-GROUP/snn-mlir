// Copyright 2026 Sensing & Control Systems, S.L.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===------------------------------ SNN op declarations --------------------------*- C++ -*-===//
#ifndef SNN_OPS_H
#define SNN_OPS_H

#include "SNN/SNNDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Pull in the auto-generated op declarations
#define GET_OP_CLASSES
#include "SNN/SNNOps.h.inc"

#endif // SNN_OPS_H
