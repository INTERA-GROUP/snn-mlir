// Copyright 2026 N Vision Systems And Technologies SL
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===- SNNOps.cpp - SNN op implementations ---------------------*- C++ -*-===//

#include "SNN/SNNOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"

// Auto-generated interface method models (SynapseOpInterface, NeuronOpInterface).
#include "SNN/SNNInterfaces.cpp.inc"

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

// True when `a` and `b` have the same rank and, dimension by dimension, the
// same static extent. Dynamic dims are accepted per dimension, for the same
// reason as above.
//
// This is what makes the neuron ops and snn.rescale rank-polymorphic. They are
// shape-preserving and elementwise, so "same shape" is their entire structural
// contract and nothing in it mentions a rank: a dense layer's vector and a conv
// layer's feature map are the same statement at different ranks. Equality is
// deliberately per-dimension rather than per-element-count, so a rank-1 state
// paired with a rank-3 input is rejected rather than silently accepted — going
// between the two is memref.collapse_shape's job, spelled out in the IR.
bool sameShape(MemRefType a, MemRefType b) {
  if (a.getRank() != b.getRank())
    return false;
  for (int64_t i = 0, e = a.getRank(); i < e; ++i) {
    int64_t da = a.getDimSize(i), db = b.getDimSize(i);
    if (ShapedType::isDynamic(da) || ShapedType::isDynamic(db))
      continue;
    if (da != db)
      return false;
  }
  return true;
}

// Shared check for the neuron ops (cubalif / lif / cubali / li). `states` are
// the in-place state memrefs (1 or 2). `spikeOutput` selects the quantized
// output contract: i8 for spiking neurons, i32 for voltage-readout neurons.
LogicalResult verifyNeuron(Operation *op, Value input, ArrayRef<Value> states,
                           Value output, bool spikeOutput) {
  auto inTy = memrefOf(input);
  auto outTy = memrefOf(output);
  Type inElem = inTy.getElementType();

  // A point neuron is elementwise and shape-preserving, at any rank: one neuron
  // per input element, wherever that element sits. So every operand must carry
  // the input's shape, and no operand is constrained to a particular rank.
  if (!sameShape(inTy, outTy))
    return op->emitOpError("expects the output to have the same shape as the "
                           "input");
  for (Value s : states)
    if (!sameShape(inTy, memrefOf(s)))
      return op->emitOpError("state operand must have the same shape as the "
                             "input");

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

  // Elementwise and shape-preserving at any rank, like the neuron ops: rescale
  // shifts each element in place and cares about scales, not geometry.
  if (!sameShape(inTy, outTy))
    return emitOpError("input and output must have the same shape");
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

// -----------------------------------------------------------------------
// SynapseOpInterface — snn.linear
// -----------------------------------------------------------------------
//
// Weights are [O, I]: dim 0 is the output count (N), dim 1 the input/reduction
// count (K). getSynapseBias() forwards to the optional bias operand (null Value
// when absent).
Value snn::LinearOp::getActivationIn() { return getInput(); }
Value snn::LinearOp::getAccumulatorOut() { return getOutput(); }
Value snn::LinearOp::getWeightMatrix() { return getWeights(); }
Value snn::LinearOp::getSynapseBias() { return getBias(); }
int64_t snn::LinearOp::getK() { return memrefOf(getWeights()).getDimSize(1); }
int64_t snn::LinearOp::getN() { return memrefOf(getWeights()).getDimSize(0); }

// -----------------------------------------------------------------------
// NeuronOpInterface — snn.cubalif / lif / li / cubali
// -----------------------------------------------------------------------
//
// The decay attribute names differ across ops: cubalif/cubali carry a separate
// cur_decay_int + vol_decay_int, whereas lif/li carry a single decay_int (the
// membrane decay). getVoltageDecay() normalizes both to "the membrane decay".

// snn.cubalif: 2-state (current, voltage), spikes.
Value snn::CubaLIFOp::getNeuronInput() { return getInput(); }
Value snn::CubaLIFOp::getNeuronOutput() { return getOutput(); }
Value snn::CubaLIFOp::getCurrentState() { return getCurrent(); }
Value snn::CubaLIFOp::getVoltageState() { return getVoltage(); }
int64_t snn::CubaLIFOp::getScale() { return getDScale(); }
int64_t snn::CubaLIFOp::getCurrentDecay() { return getCurDecayInt(); }
int64_t snn::CubaLIFOp::getVoltageDecay() { return getVolDecayInt(); }
int64_t snn::CubaLIFOp::getThreshold() { return getThresholdInt(); }
int64_t snn::CubaLIFOp::getReset() { return 0; }
bool snn::CubaLIFOp::hasCurrentStage() { return true; }
bool snn::CubaLIFOp::producesSpike() { return true; }

// snn.lif: 1-state (voltage), spikes. Single decay_int = membrane decay.
Value snn::LIFOp::getNeuronInput() { return getInput(); }
Value snn::LIFOp::getNeuronOutput() { return getOutput(); }
Value snn::LIFOp::getCurrentState() { return Value(); }
Value snn::LIFOp::getVoltageState() { return getVoltage(); }
int64_t snn::LIFOp::getScale() { return getDScale(); }
int64_t snn::LIFOp::getCurrentDecay() { return 0; }
int64_t snn::LIFOp::getVoltageDecay() { return getDecayInt(); }
int64_t snn::LIFOp::getThreshold() { return getThresholdInt(); }
int64_t snn::LIFOp::getReset() { return getVResetInt(); }
bool snn::LIFOp::hasCurrentStage() { return false; }
bool snn::LIFOp::producesSpike() { return true; }

// snn.li: 1-state (voltage), voltage readout (no spike, no threshold).
Value snn::LIOp::getNeuronInput() { return getInput(); }
Value snn::LIOp::getNeuronOutput() { return getOutput(); }
Value snn::LIOp::getCurrentState() { return Value(); }
Value snn::LIOp::getVoltageState() { return getVoltage(); }
int64_t snn::LIOp::getScale() { return getDScale(); }
int64_t snn::LIOp::getCurrentDecay() { return 0; }
int64_t snn::LIOp::getVoltageDecay() { return getDecayInt(); }
int64_t snn::LIOp::getThreshold() { return 0; }
int64_t snn::LIOp::getReset() { return 0; }
bool snn::LIOp::hasCurrentStage() { return false; }
bool snn::LIOp::producesSpike() { return false; }

// snn.cubali: 2-state (current, voltage), voltage readout (no spike/threshold).
Value snn::CubaLIOp::getNeuronInput() { return getInput(); }
Value snn::CubaLIOp::getNeuronOutput() { return getOutput(); }
Value snn::CubaLIOp::getCurrentState() { return getCurrent(); }
Value snn::CubaLIOp::getVoltageState() { return getVoltage(); }
int64_t snn::CubaLIOp::getScale() { return getDScale(); }
int64_t snn::CubaLIOp::getCurrentDecay() { return getCurDecayInt(); }
int64_t snn::CubaLIOp::getVoltageDecay() { return getVolDecayInt(); }
int64_t snn::CubaLIOp::getThreshold() { return 0; }
int64_t snn::CubaLIOp::getReset() { return 0; }
bool snn::CubaLIOp::hasCurrentStage() { return true; }
bool snn::CubaLIOp::producesSpike() { return false; }
