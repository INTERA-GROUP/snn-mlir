# Implementing a new lowering pass

## What is a lowering pass?

In MLIR, a **lowering pass** rewrites operations from one dialect into operations of a
lower-level dialect — a step on the journey from abstract IR to executable code. snn-mlir ships
one reference lowering, `SNNToLinalg`, which converts each `snn` op into standard
`linalg`/`arith` operations that any MLIR-based CPU backend can consume.

This is the extension point **for hardware developers**. To target your own accelerator, you
write a pass that lowers the `snn` ops to *your* representation — whether that's another MLIR
dialect, intrinsics, or calls into a hardware runtime. The same `network.mlir` can then be
compiled for a custom **FPGA** target or an **ASIC** implementation without touching the
frontend or the dialect definition. `lib/Conversion/SNNToLinalg/SNNToLinalg.cpp` is the
reference implementation to copy from.

## 1. Create the pass files

```
include/SNN/Conversion/SNNToMyBackend.h
lib/Conversion/SNNToMyBackend/SNNToMyBackend.cpp
lib/Conversion/SNNToMyBackend/CMakeLists.txt
```

## 2. Declare your pass in the header

```cpp
#include "mlir/Pass/Pass.h"
#include <memory>

namespace snn {
  std::unique_ptr<mlir::Pass> createConvertSNNToMyBackendPass();
  void registerConvertSNNToMyBackendPass();
} // namespace snn
```

## 3. Implement a rewrite pattern per op

```cpp
#include "SNN/SNNOps.h"

struct LowerLinear : public OpRewritePattern<snn::LinearOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(snn::LinearOp op,
                                PatternRewriter &rewriter) const override {
    // Replace op with your backend calls
    rewriter.eraseOp(op);
    return success();
  }
};
```

## 4. Wire up the pass

```cpp
struct ConvertSNNToMyBackendPass
    : public PassWrapper<ConvertSNNToMyBackendPass, OperationPass<ModuleOp>> {

  StringRef getArgument() const override { return "convert-snn-to-mybackend"; }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerLinear, LowerRescale, LowerCubaLIF>(&getContext());

    ConversionTarget target(getContext());
    target.addIllegalDialect<snn::SNNDialect>();
    target.addLegalDialect</* your dialects */>();

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};
```

## 5. Register in CMake

Use `add_mlir_conversion_library()` — see `lib/Conversion/SNNToLinalg/CMakeLists.txt` as a
template.

## Reading a layer through op interfaces

Matching a concrete op type (`snn::LinearOp`, `snn::CubaLIFOp`, …) is the right tool when your
lowering is *op-specific*. But most backends want to treat "a synapse layer" or "a spiking
neuron" **uniformly** — without a `switch` over the four neuron kinds, and without depending on
the exact attribute names each op happens to use. Two op interfaces let you do that. They are
declared **natively** on the ops, so every `snn.linear` *is* a `SynapseOpInterface` and every one
of the four neuron ops *is* a `NeuronOpInterface` — no registration, nothing to attach.

### `SynapseOpInterface` — `snn.linear`

A dense layer computing `accumulator = weights @ input`. It exposes the operands and the shape a
backend needs without knowing the concrete op:

| Method | Returns |
|---|---|
| `getActivationIn()` | input activation vector (the matmul's LHS) |
| `getWeightMatrix()` | constant weight matrix `[N, K]` (the matmul's RHS) |
| `getAccumulatorOut()` | accumulator destination (the raw dot-product output) |
| `getSynapseBias()` | optional bias vector, or a null `Value` when the op has none |
| `getK()` | number of inputs — the reduction dimension |
| `getN()` | number of output neurons |

### `NeuronOpInterface` — `snn.cubalif` / `snn.lif` / `snn.li` / `snn.cubali`

A point-neuron whose integrate-and-fire dynamics you can read without knowing which of the four
ops it is:

| Method | Returns |
|---|---|
| `getNeuronInput()` / `getNeuronOutput()` | input current / output (spikes or voltage) |
| `getCurrentState()` / `getVoltageState()` | the in-place state memrefs |
| `getScale()` | fixed-point scale (`d_scale`); decays/threshold are scaled by `1 << scale` |
| `getCurrentDecay()` / `getVoltageDecay()` | fixed-point synaptic-current / membrane decays |
| `getThreshold()` / `getReset()` | fixed-point firing threshold / post-spike voltage reset |
| `hasCurrentStage()` / `producesSpike()` | capability predicates (see below) |

The two predicates collapse the four kinds into one uniform reader:

| op | `hasCurrentStage()` | `producesSpike()` |
|---|:---:|:---:|
| `snn.cubalif` | ✅ | ✅ |
| `snn.lif` | — | ✅ |
| `snn.li` | — | — |
| `snn.cubali` | ✅ | — |

Three consequences to honor:

- `getCurrentState()` is a null `Value` for the single-state neurons (`lif`, `li`) — guard it with
  `hasCurrentStage()`.
- `getThreshold()` / `getReset()` are meaningful only when `producesSpike()`; the voltage-readout
  neurons (`li`, `cubali`) have no threshold and report `0`.
- `getVoltageDecay()` normalizes the differing decay attribute names — `cubalif`/`cubali` carry a
  separate current and voltage decay, `lif`/`li` carry a single membrane decay — to "the membrane
  decay" for all four.

### Matching on the interface

Write one pattern over the interface instead of four over the concrete ops:

```cpp
#include "SNN/SNNInterfaces.h"

struct LowerNeuron : public OpInterfaceRewritePattern<snn::NeuronOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(snn::NeuronOpInterface neuron,
                                PatternRewriter &rewriter) const override {
    Value in    = neuron.getNeuronInput();
    Value volt  = neuron.getVoltageState();
    int64_t vd  = neuron.getVoltageDecay();

    if (neuron.hasCurrentStage()) {
      // integrate neuron.getCurrentState() with neuron.getCurrentDecay()
    }
    if (neuron.producesSpike()) {
      // compare the membrane potential against neuron.getThreshold(),
      // then reset it to neuron.getReset()
    }

    // ... emit your backend ops ...
    rewriter.eraseOp(neuron);
    return success();
  }
};
```

Include `SNN/SNNInterfaces.h` and link `MLIRSNN` — the interfaces live in the dialect library, so
there is nothing extra to register or build. `SynapseOpInterface` is used the same way with
`OpInterfaceRewritePattern<snn::SynapseOpInterface>`.

> **When to prefer which.** Reach for the interfaces whenever the lowering logic is the same shape
> across ops and only the parameters differ — that is the common case for a hardware backend, and
> it means adding a fifth neuron op later costs you nothing here. Keep a concrete-op pattern when a
> single op needs genuinely bespoke handling.
