// Copyright 2026 N Vision Systems And Technologies SL
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===- SNNInterfaces.td - SNN op interfaces ----------------*- tablegen -*-===//
//
// OpInterfaces that let a consumer (a lowering, an analysis, a backend) read an
// SNN layer uniformly, without switching on the concrete op type:
//
//   SynapseOpInterface — a weight-matrix layer (snn.linear): the operands and
//                        shape of `accumulator = weights @ input`.
//   NeuronOpInterface  — an SNN point-neuron (snn.cubalif / lif / li / cubali):
//                        its operands, fixed-point parameters, and the two
//                        capability predicates that distinguish the four kinds.
//
//===----------------------------------------------------------------------===//

#ifndef SNN_INTERFACES_TD
#define SNN_INTERFACES_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// SynapseOpInterface
//===----------------------------------------------------------------------===//

def SynapseOpInterface : OpInterface<"SynapseOpInterface"> {
  let description = [{
    A dense linear layer computing `accumulator = weights @ input`. Exposes the
    operands and the K/N shape a consumer needs without knowing the concrete op.

    The weight matrix is `[N, K]` (output-by-input, the NIR/PyTorch convention):
    `getN()` is the number of output neurons, `getK()` the number of inputs (the
    reduction dimension).

    Method names are deliberately distinct from the ops' own operand accessors
    (e.g. `getSynapseBias` vs snn.linear's `getBias`) so the interface can be
    declared natively on the op without redeclaring a generated accessor.
  }];

  let cppNamespace = "::snn";

  let methods = [
    InterfaceMethod<
      /*desc=*/"The input activation vector (the matmul's LHS).",
      /*retTy=*/"::mlir::Value", /*methodName=*/"getActivationIn">,
    InterfaceMethod<
      /*desc=*/"The accumulator destination (the raw dot-product output).",
      /*retTy=*/"::mlir::Value", /*methodName=*/"getAccumulatorOut">,
    InterfaceMethod<
      /*desc=*/"The constant weight matrix (the matmul's RHS).",
      /*retTy=*/"::mlir::Value", /*methodName=*/"getWeightMatrix">,
    InterfaceMethod<
      /*desc=*/"The optional bias vector, or a null Value when the op has none. "
               "Length is N (one per output neuron).",
      /*retTy=*/"::mlir::Value", /*methodName=*/"getSynapseBias">,
    InterfaceMethod<
      /*desc=*/"Number of input neurons (the reduction dim K).",
      /*retTy=*/"int64_t", /*methodName=*/"getK">,
    InterfaceMethod<
      /*desc=*/"Number of output neurons (N).",
      /*retTy=*/"int64_t", /*methodName=*/"getN">,
  ];
}

//===----------------------------------------------------------------------===//
// NeuronOpInterface
//===----------------------------------------------------------------------===//

def NeuronOpInterface : OpInterface<"NeuronOpInterface"> {
  let description = [{
    An SNN point-neuron whose integrate-and-fire dynamics a consumer can read
    uniformly, without knowing whether the op is an snn.cubalif / lif / li /
    cubali. Exposes the operands, the Q-format fixed-point parameters, and two
    capability predicates:

      op        hasCurrentStage()   producesSpike()
      cubalif        yes                 yes
      lif            no                  yes
      li             no                  no
      cubali         yes                 no

    `getCurrentState()` is null for the single-state neurons (lif, li) — guard it
    with `hasCurrentStage()`. `getThresholdInt()` / `getVResetInt()` are
    meaningful only when `producesSpike()` (li / cubali have no threshold).

    The decay attribute names differ across ops (cubalif/cubali carry a separate
    current and voltage decay; lif/li carry a single membrane decay);
    `getVoltageDecay()` normalizes all of them to "the membrane decay".

    The scalar getters return the fixed-point (quantized) integer form and have
    names distinct from the ops' own attribute accessors (e.g. `getScale` vs
    `getDScale`) so the interface can be declared natively on the ops without
    redeclaring a generated accessor.
  }];

  let cppNamespace = "::snn";

  let methods = [
    InterfaceMethod<
      /*desc=*/"The input current the neuron integrates.",
      /*retTy=*/"::mlir::Value", /*methodName=*/"getNeuronInput">,
    InterfaceMethod<
      /*desc=*/"The neuron output (i8 spikes if producesSpike(), else i32 voltage).",
      /*retTy=*/"::mlir::Value", /*methodName=*/"getNeuronOutput">,
    InterfaceMethod<
      /*desc=*/"The synaptic-current state, updated in place; null if !hasCurrentStage().",
      /*retTy=*/"::mlir::Value", /*methodName=*/"getCurrentState">,
    InterfaceMethod<
      /*desc=*/"The membrane-voltage state, updated in place (present on all neurons).",
      /*retTy=*/"::mlir::Value", /*methodName=*/"getVoltageState">,
    InterfaceMethod<
      /*desc=*/"Fixed-point scale (d_scale); decays/threshold are scaled by 1<<scale.",
      /*retTy=*/"int64_t", /*methodName=*/"getScale">,
    InterfaceMethod<
      /*desc=*/"Fixed-point synaptic-current decay; 0 when !hasCurrentStage().",
      /*retTy=*/"int64_t", /*methodName=*/"getCurrentDecay">,
    InterfaceMethod<
      /*desc=*/"Fixed-point membrane-voltage decay (present on all neurons).",
      /*retTy=*/"int64_t", /*methodName=*/"getVoltageDecay">,
    InterfaceMethod<
      /*desc=*/"Fixed-point firing threshold; meaningful only when producesSpike().",
      /*retTy=*/"int64_t", /*methodName=*/"getThreshold">,
    InterfaceMethod<
      /*desc=*/"Fixed-point voltage reset value after a spike; meaningful only "
               "when producesSpike().",
      /*retTy=*/"int64_t", /*methodName=*/"getReset">,
    InterfaceMethod<
      /*desc=*/"True iff the neuron has a synaptic-current stage.",
      /*retTy=*/"bool", /*methodName=*/"hasCurrentStage">,
    InterfaceMethod<
      /*desc=*/"True iff the neuron emits spikes rather than a voltage readout.",
      /*retTy=*/"bool", /*methodName=*/"producesSpike">,
  ];
}

#endif // SNN_INTERFACES_TD
