// Copyright 2026 N Vision Systems And Technologies SL
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===- ----------------- SNN dialect definition -------------*- tablegen -*-===//
//
// Defines the SNN (Spiking Neural Network) dialect.
// Compatible with NIR (Neuromorphic Intermediate Representation) v1.0.7.
//
//===----------------------------------------------------------------------===//

#ifndef SNN_DIALECT_TD
#define SNN_DIALECT_TD

include "mlir/IR/OpBase.td"

def SNN_Dialect : Dialect {
  let name = "snn";
  let summary = "Spiking Neural Network operations (NIR-compatible)";
  let description = [{
    The SNN dialect provides high-level operations for spiking neural
    networks. Ops are type-polymorphic: they work with both float
    (f32 weights/state) and quantized (i8 weights, i32 state) types.
    Lowering passes inspect the concrete types to emit the appropriate
    arithmetic (float vs fixed-point).
  }];
  let cppNamespace = "::snn";
}

#endif // SNN_DIALECT_TD
