// Copyright 2026 N Vision Systems And Technologies SL
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===- SNNOps.cpp - SNN op implementations ---------------------*- C++ -*-===//

#include "SNN/SNNOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"

// Pull in the auto-generated op method bodies (builders, parsers, printers,
// verifiers, etc. — everything derived from the assemblyFormat and traits
// you declared in SNNOps.td).
#define GET_OP_CLASSES
#include "SNN/SNNOps.cpp.inc"

using namespace mlir;

// -----------------------------------------------------------------------
// Verifier helpers
// -----------------------------------------------------------------------
//
// SNN ops are type-polymorphic on a single axis: the *input* element type
// decides the mode.
//
//   • float mode     — input is any FloatType; every other operand must
//                       share that exact float type.
//   • quantized mode — input is an integer; element types are locked to the
//                       i8/i32 contract that the quantizer and downstream
//                       backends rely on. Rejecting anything else here turns
//                       an unwritten assumption into a checked diagnostic.

namespace {

MemRefType memrefOf(Value v) { return cast<MemRefType>(v.getType()); }

// True when `t` is 1-D and (statically) the same length as `n`. Dynamic dims
// are accepted — shape inconsistency there is not something we can prove here.
bool sameLength(MemRefType t, int64_t n) {
  if (t.getRank() != 1)
    return false;
  if (ShapedType::isDynamic(n) || t.isDynamicDim(0))
    return true;
  return t.getDimSize(0) == n;
}

// Shared check for the neuron ops (cubalif / lif / cubali / li). `states` are
// the in-place state memrefs (1 or 2). `spikeOutput` selects the quantized
// output contract: i8 for spiking neurons, i32 for voltage-readout neurons.
LogicalResult verifyNeuron(Operation *op, Value input, ArrayRef<Value> states,
                           Value output, bool spikeOutput) {
  auto inTy = memrefOf(input);
  auto outTy = memrefOf(output);
  Type inElem = inTy.getElementType();

  // All operands must be 1-D vectors of equal length.
  int64_t n = inTy.getRank() == 1 ? inTy.getDimSize(0) : ShapedType::kDynamic;
  if (inTy.getRank() != 1 || !sameLength(outTy, n))
    return op->emitOpError("expects all operands to be 1-D of equal length");
  for (Value s : states)
    if (!sameLength(memrefOf(s), n))
      return op->emitOpError("state operand length must match the input length");

  if (isa<FloatType>(inElem)) {
    for (Value s : states)
      if (memrefOf(s).getElementType() != inElem)
        return op->emitOpError(
            "float mode requires state to share the input float type");
    if (outTy.getElementType() != inElem)
      return op->emitOpError(
          "float mode requires output to share the input float type");
  } else if (inElem.isInteger(32)) {
    for (Value s : states)
      if (!memrefOf(s).getElementType().isInteger(32))
        return op->emitOpError("quantized mode requires i32 state");
    bool okOut = spikeOutput ? outTy.getElementType().isInteger(8)
                             : outTy.getElementType().isInteger(32);
    if (!okOut)
      return op->emitOpError("quantized mode requires ")
             << (spikeOutput ? "i8 spike output" : "i32 voltage output");
  } else {
    return op->emitOpError(
        "input element type must be a float or i32 (quantized)");
  }
  return success();
}

} // namespace

// -----------------------------------------------------------------------
// snn.linear
// -----------------------------------------------------------------------
LogicalResult snn::LinearOp::verify() {
  auto inTy = memrefOf(getInput());
  auto wTy = memrefOf(getWeights());
  auto outTy = memrefOf(getOutput());

  if (inTy.getRank() != 1 || outTy.getRank() != 1 || wTy.getRank() != 2)
    return emitOpError("expects 1-D input/output and 2-D weights");

  // weights are [output_size, input_size].
  int64_t I = inTy.getDimSize(0);
  int64_t O = outTy.getDimSize(0);
  if (!inTy.isDynamicDim(0) && !wTy.isDynamicDim(1) && wTy.getDimSize(1) != I)
    return emitOpError("weights inner dim (") << wTy.getDimSize(1)
           << ") must match input size (" << I << ")";
  if (!outTy.isDynamicDim(0) && !wTy.isDynamicDim(0) && wTy.getDimSize(0) != O)
    return emitOpError("weights outer dim (") << wTy.getDimSize(0)
           << ") must match output size (" << O << ")";

  Type inElem = inTy.getElementType();
  if (isa<FloatType>(inElem)) {
    if (wTy.getElementType() != inElem || outTy.getElementType() != inElem)
      return emitOpError(
          "float mode requires input, weights, and output to share the same "
          "float element type");
    if (getBias() && memrefOf(getBias()).getElementType() != inElem)
      return emitOpError("float mode bias must share the float element type");
  } else if (inElem.isInteger(8)) {
    if (!wTy.getElementType().isInteger(8))
      return emitOpError("quantized mode requires i8 weights");
    if (!outTy.getElementType().isInteger(32))
      return emitOpError("quantized mode requires i32 output");
    if (getBias() && !memrefOf(getBias()).getElementType().isInteger(32))
      return emitOpError("quantized mode bias must be i32");
  } else {
    return emitOpError("input element type must be a float or i8 (quantized)");
  }

  if (getBias() && !sameLength(memrefOf(getBias()), O))
    return emitOpError("bias length must match the output size");
  return success();
}

// -----------------------------------------------------------------------
// snn.rescale  (quantized-only)
// -----------------------------------------------------------------------
LogicalResult snn::RescaleOp::verify() {
  auto inTy = memrefOf(getInput());
  auto outTy = memrefOf(getOutput());

  if (!isa<IntegerType>(inTy.getElementType()))
    return emitOpError("operates on integer (quantized) memrefs only");
  if (!outTy.getElementType().isInteger(32))
    return emitOpError("output must be i32");

  int64_t n = inTy.getRank() == 1 ? inTy.getDimSize(0) : ShapedType::kDynamic;
  if (inTy.getRank() != 1 || !sameLength(outTy, n))
    return emitOpError("input and output must be 1-D of equal length");
  return success();
}

// -----------------------------------------------------------------------
// Neuron ops
// -----------------------------------------------------------------------
LogicalResult snn::CubaLIFOp::verify() {
  return verifyNeuron(getOperation(), getInput(), {getCurrent(), getVoltage()},
                      getOutput(), /*spikeOutput=*/true);
}

LogicalResult snn::LIFOp::verify() {
  return verifyNeuron(getOperation(), getInput(), {getVoltage()}, getOutput(),
                      /*spikeOutput=*/true);
}

LogicalResult snn::CubaLIOp::verify() {
  return verifyNeuron(getOperation(), getInput(), {getCurrent(), getVoltage()},
                      getOutput(), /*spikeOutput=*/false);
}

LogicalResult snn::LIOp::verify() {
  return verifyNeuron(getOperation(), getInput(), {getVoltage()}, getOutput(),
                      /*spikeOutput=*/false);
}
