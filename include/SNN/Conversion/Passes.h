// SPDX-License-Identifier: Apache-2.0
//===- Passes.h - SNN conversion pass registration ------------*- C++ -*-===//
#ifndef SNN_CONVERSION_PASSES_H
#define SNN_CONVERSION_PASSES_H

#include "SNN/Conversion/SNNToLinalg.h"
#include "mlir/Pass/Pass.h"

namespace snn {

// Pull in the auto-generated pass + option declarations derived from Passes.td.
// The pass base class is emitted in SNNToLinalg.cpp via GEN_PASS_DEF; the
// constructor (createConvertSNNToLinalgPass) and the registration hook
// (registerConvertSNNToLinalgPass) are declared in SNNToLinalg.h and defined
// in SNNToLinalg.cpp, so the library exports them as normal linkable symbols.
#define GEN_PASS_DECL
#include "SNN/Conversion/Passes.h.inc"

} // namespace snn

#endif // SNN_CONVERSION_PASSES_H
