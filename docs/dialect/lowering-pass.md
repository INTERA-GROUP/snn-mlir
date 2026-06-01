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
