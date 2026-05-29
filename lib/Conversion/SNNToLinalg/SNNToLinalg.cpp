// SPDX-License-Identifier: Apache-2.0
//
// Lowers SNN dialect ops to linalg/arith for CPU execution.
// Dispatches on operand element types: float → linalg.matvec/mulf,
// integer → linalg.generic with extsi/muli.

#include "SNN/SNNOps.h"
#include "SNN/Conversion/SNNToLinalg.h"
#include "SNN/Conversion/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
//  Helper: is this element type floating point?
//===----------------------------------------------------------------------===//
static bool isFloatMemRef(Value v) {
  return isa<FloatType>(cast<MemRefType>(v.getType()).getElementType());
}

//===----------------------------------------------------------------------===//
//  Pattern: snn.linear → linalg.matvec (float) or linalg.generic (int)
//      Uses linalg.generic since w=i8; in=i8; out=i16/i32
//===----------------------------------------------------------------------===//
struct LowerLinear : public OpRewritePattern<snn::LinearOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(snn::LinearOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value weights = op.getWeights();
    Value output = op.getOutput();

    auto outTy = cast<MemRefType>(output.getType());
    Type elemTy = outTy.getElementType();

    // Zero-fill output
    Value zero;
    if (isFloatMemRef(output))
      zero = arith::ConstantOp::create(rewriter,
          loc, elemTy, rewriter.getFloatAttr(elemTy, 0.0));
    else
      zero = arith::ConstantOp::create(rewriter,
          loc, elemTy, rewriter.getIntegerAttr(elemTy, 0));
    linalg::FillOp::create(rewriter, loc, zero, output);

    if (isFloatMemRef(input)) {
      // Float path: linalg.matvec
      linalg::MatvecOp::create(rewriter,
          loc, ValueRange{weights, input}, ValueRange{output});
    } else {
      // Quantized path: linalg.generic with sign-extension
      SmallVector<AffineMap> maps = {
          AffineMap::get(2, 0, {rewriter.getAffineDimExpr(0),
                                rewriter.getAffineDimExpr(1)},
                         rewriter.getContext()), // weights[i,j]
          AffineMap::get(2, 0, {rewriter.getAffineDimExpr(1)},
                         rewriter.getContext()), // input[j]
          AffineMap::get(2, 0, {rewriter.getAffineDimExpr(0)},
                         rewriter.getContext()), // output[i]
      };
      SmallVector<utils::IteratorType> iterTypes = {
          utils::IteratorType::parallel,
          utils::IteratorType::reduction,
      };

      Type accTy = outTy.getElementType(); // i16 or i32

      linalg::GenericOp::create(rewriter,
          loc, TypeRange{}, ValueRange{weights, input}, ValueRange{output},
          maps, iterTypes,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value w = args[0], x = args[1], acc = args[2];
            Value wExt = arith::ExtSIOp::create(b, loc, accTy, w);
            Value xExt = arith::ExtSIOp::create(b, loc, accTy, x);
            Value prod = arith::MulIOp::create(b, loc, wExt, xExt);
            Value sum  = arith::AddIOp::create(b, loc, acc, prod);
            linalg::YieldOp::create(b, loc, sum);
          });
    }

    // Optional bias add: output[i] += bias[i]
    Value bias = op.getBias();
    if (bias) {
      AffineMap id = rewriter.getDimIdentityMap();
      SmallVector<AffineMap> biasMaps = {id, id};
      SmallVector<utils::IteratorType> biasIter = {utils::IteratorType::parallel};

      if (isFloatMemRef(output)) {
        linalg::GenericOp::create(rewriter,
            loc, TypeRange{}, ValueRange{bias}, ValueRange{output},
            biasMaps, biasIter,
            [&](OpBuilder &b, Location loc, ValueRange args) {
              Value bval = args[0], acc = args[1];
              Value sum = arith::AddFOp::create(b, loc, acc, bval);
              linalg::YieldOp::create(b, loc, sum);
            });
      } else {
        // Bias is i32 (same scale as the MAC accumulator — no conversion needed).
        linalg::GenericOp::create(rewriter,
            loc, TypeRange{}, ValueRange{bias}, ValueRange{output},
            biasMaps, biasIter,
            [&](OpBuilder &b, Location loc, ValueRange args) {
              Value bval = args[0], acc = args[1];
              Value sum  = arith::AddIOp::create(b, loc, acc, bval);
              linalg::YieldOp::create(b, loc, sum);
            });
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
//  Pattern: snn.rescale → linalg.generic with extsi + shli/shrsi
//===----------------------------------------------------------------------===//
struct LowerRescale : public OpRewritePattern<snn::RescaleOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(snn::RescaleOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value output = op.getOutput();

    int64_t wScale = op.getWScale();
    int64_t dScale = op.getDScale();
    int64_t shift = dScale - wScale;

    auto inTy = cast<MemRefType>(input.getType());
    Type inElem = inTy.getElementType();
    auto outTy = cast<MemRefType>(output.getType());
    Type outElem = outTy.getElementType(); // i32

    AffineMap id = rewriter.getDimIdentityMap();
    SmallVector<AffineMap> maps = {id, id};
    SmallVector<utils::IteratorType> iterTypes = {
        utils::IteratorType::parallel};

    linalg::GenericOp::create(rewriter,
        loc, TypeRange{}, ValueRange{input}, ValueRange{output}, maps,
        iterTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value val = args[0];

          // Sign-extend if narrower than output
          if (inElem != outElem)
            val = arith::ExtSIOp::create(b, loc, outElem, val);

          // Shift to align scales
          if (shift > 0) {
            Value shiftVal = arith::ConstantOp::create(b,
                loc, outElem, b.getIntegerAttr(outElem, shift));
            val = arith::ShLIOp::create(b, loc, val, shiftVal);
          } else if (shift < 0) {
            Value shiftVal = arith::ConstantOp::create(b,
                loc, outElem, b.getIntegerAttr(outElem, -shift));
            val = arith::ShRSIOp::create(b, loc, val, shiftVal);
          }

          linalg::YieldOp::create(b, loc, val);
        });

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
//  Pattern: snn.cubalif → linalg.generic (float or quantized dynamics)
//===----------------------------------------------------------------------===//
struct LowerCubaLIF : public OpRewritePattern<snn::CubaLIFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(snn::CubaLIFOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value current = op.getCurrent();
    Value voltage = op.getVoltage();
    Value output = op.getOutput();

    AffineMap id = rewriter.getDimIdentityMap();
    SmallVector<AffineMap> maps = {id, id, id, id};
    SmallVector<utils::IteratorType> iterTypes = {
        utils::IteratorType::parallel};

    if (isFloatMemRef(input)) {
      // Float path
      double curDecay = op.getCurDecayFloat().convertToDouble();
      double volDecay = op.getVolDecayFloat().convertToDouble();
      double threshold = op.getThresholdFloat().convertToDouble();

      // Derive the float element type from the operand so any FloatType
      // (f16/bf16/f32/f64) lowers correctly — not just f32.
      Type fTy = cast<MemRefType>(input.getType()).getElementType();

      linalg::GenericOp::create(rewriter,
          loc, TypeRange{}, ValueRange{input},
          ValueRange{current, voltage, output}, maps, iterTypes,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value s = args[0], c = args[1], v = args[2];
            // current = cur_decay * current + input
            Value cd = arith::ConstantOp::create(b,
                loc, fTy, b.getFloatAttr(fTy, curDecay));
            Value cScaled = arith::MulFOp::create(b, loc, cd, c);
            Value cNew    = arith::AddFOp::create(b, loc, cScaled, s);
            // voltage = vol_decay * voltage + current
            Value vd = arith::ConstantOp::create(b,
                loc, fTy, b.getFloatAttr(fTy, volDecay));
            Value vScaled = arith::MulFOp::create(b, loc, vd, v);
            Value vNew    = arith::AddFOp::create(b, loc, vScaled, cNew);
            // spike = voltage > threshold
            Value th    = arith::ConstantOp::create(b, loc, fTy, b.getFloatAttr(fTy, threshold));
            Value fzero = arith::ConstantOp::create(b, loc, fTy, b.getFloatAttr(fTy, 0.0));
            Value fone  = arith::ConstantOp::create(b, loc, fTy, b.getFloatAttr(fTy, 1.0));
            Value cmp   = arith::CmpFOp::create(b,
                loc, arith::CmpFPredicate::OGT, vNew, th);
            Value spike  = arith::SelectOp::create(b, loc, cmp, fone, fzero);
            Value vFinal = arith::SelectOp::create(b, loc, cmp, fzero, vNew);
            linalg::YieldOp::create(b, loc, ValueRange{cNew, vFinal, spike});
          });
    } else {
      // Quantized path (Q12)
      int64_t dScale      = op.getDScale();
      int64_t curDecayInt = op.getCurDecayInt();
      int64_t volDecayInt = op.getVolDecayInt();
      int64_t thresholdInt = op.getThresholdInt();

      // i32 throughout: matches RV32 target; Q12 dynamics bound state below overflow.
      Type i32 = rewriter.getI32Type();
      auto outElem =
          cast<MemRefType>(output.getType()).getElementType(); // i8

      linalg::GenericOp::create(rewriter,
          loc, TypeRange{}, ValueRange{input},
          ValueRange{current, voltage, output}, maps, iterTypes,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value s = args[0], c = args[1], v = args[2];

            Value shiftVal = arith::ConstantOp::create(b,
                loc, i32, b.getI32IntegerAttr(dScale));

            // c_new = (cur_decay * c) >> d_scale + input
            Value cd      = arith::ConstantOp::create(b, loc, i32, b.getI32IntegerAttr(curDecayInt));
            Value cProd   = arith::MulIOp::create(b, loc, cd, c);
            Value cShifted = arith::ShRSIOp::create(b, loc, cProd, shiftVal);
            Value cNew    = arith::AddIOp::create(b, loc, cShifted, s);

            // v_new = (vol_decay * v) >> d_scale + c_new
            Value vd      = arith::ConstantOp::create(b, loc, i32, b.getI32IntegerAttr(volDecayInt));
            Value vProd   = arith::MulIOp::create(b, loc, vd, v);
            Value vShifted = arith::ShRSIOp::create(b, loc, vProd, shiftVal);
            Value vNew    = arith::AddIOp::create(b, loc, vShifted, cNew);
            // spike = v_new > threshold ? 1 : 0
            Value th    = arith::ConstantOp::create(b, loc, i32, b.getI32IntegerAttr(thresholdInt));
            Value cmp   = arith::CmpIOp::create(b,
                loc, arith::CmpIPredicate::sgt, vNew, th);
            Value izero  = arith::ConstantOp::create(b, loc, i32, b.getI32IntegerAttr(0));
            Value vFinal = arith::SelectOp::create(b, loc, cmp, izero, vNew);
            // spike output (i8: 0 or 1)
            Value one8  = arith::ConstantOp::create(b, loc, outElem, b.getIntegerAttr(outElem, 1));
            Value zero8 = arith::ConstantOp::create(b, loc, outElem, b.getIntegerAttr(outElem, 0));
            Value spike = arith::SelectOp::create(b, loc, cmp, one8, zero8);
            linalg::YieldOp::create(b, loc, ValueRange{cNew, vFinal, spike});
          });
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
//  Pattern: snn.lif → linalg.generic (float or quantized dynamics)
//      1-state (voltage), spike output (f32 or i8)
//===----------------------------------------------------------------------===//
struct LowerLIF : public OpRewritePattern<snn::LIFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(snn::LIFOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value voltage = op.getVoltage();
    Value output = op.getOutput();

    AffineMap id = rewriter.getDimIdentityMap();
    SmallVector<AffineMap> maps = {id, id, id};
    SmallVector<utils::IteratorType> iterTypes = {utils::IteratorType::parallel};

    if (isFloatMemRef(input)) {
      double decay     = op.getDecayFloat().convertToDouble();
      double threshold = op.getThresholdFloat().convertToDouble();
      double vReset    = op.getVResetFloat().convertToDouble();
      Type fTy = cast<MemRefType>(input.getType()).getElementType();

      linalg::GenericOp::create(rewriter,
          loc, TypeRange{}, ValueRange{input},
          ValueRange{voltage, output}, maps, iterTypes,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value s = args[0], v = args[1];
            // voltage = decay * voltage + input
            Value dc      = arith::ConstantOp::create(b, loc, fTy, b.getFloatAttr(fTy, decay));
            Value vScaled = arith::MulFOp::create(b, loc, dc, v);
            Value vNew    = arith::AddFOp::create(b, loc, vScaled, s);
            // spike = vNew > threshold (snntorch fires on strict >)
            Value th    = arith::ConstantOp::create(b, loc, fTy, b.getFloatAttr(fTy, threshold));
            Value fzero = arith::ConstantOp::create(b, loc, fTy, b.getFloatAttr(fTy, 0.0));
            Value fone  = arith::ConstantOp::create(b, loc, fTy, b.getFloatAttr(fTy, 1.0));
            Value cmp   = arith::CmpFOp::create(b,
                loc, arith::CmpFPredicate::OGT, vNew, th);
            Value spike  = arith::SelectOp::create(b, loc, cmp, fone, fzero);
            Value vr     = arith::ConstantOp::create(b, loc, fTy, b.getFloatAttr(fTy, vReset));
            Value vFinal = arith::SelectOp::create(b, loc, cmp, vr, vNew);
            linalg::YieldOp::create(b, loc, ValueRange{vFinal, spike});
          });
    } else {
      // Quantized path (Q12)
      int64_t dScale      = op.getDScale();
      int64_t decayInt    = op.getDecayInt();
      int64_t thresholdInt = op.getThresholdInt();
      int64_t vResetInt   = op.getVResetInt();

      Type i32 = rewriter.getI32Type();
      auto outElem =
          cast<MemRefType>(output.getType()).getElementType(); // i8

      linalg::GenericOp::create(rewriter,
          loc, TypeRange{}, ValueRange{input},
          ValueRange{voltage, output}, maps, iterTypes,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value s = args[0], v = args[1];

            Value shiftVal = arith::ConstantOp::create(b,
                loc, i32, b.getI32IntegerAttr(dScale));

            // voltage = (decay * v) >> d_scale + input
            Value dc      = arith::ConstantOp::create(b, loc, i32, b.getI32IntegerAttr(decayInt));
            Value vProd   = arith::MulIOp::create(b, loc, dc, v);
            Value vShifted = arith::ShRSIOp::create(b, loc, vProd, shiftVal);
            Value vNew    = arith::AddIOp::create(b, loc, vShifted, s);
            // spike = vNew > threshold_int
            Value th    = arith::ConstantOp::create(b, loc, i32, b.getI32IntegerAttr(thresholdInt));
            Value cmp   = arith::CmpIOp::create(b,
                loc, arith::CmpIPredicate::sgt, vNew, th);
            Value vr     = arith::ConstantOp::create(b, loc, i32, b.getI32IntegerAttr(vResetInt));
            Value vFinal = arith::SelectOp::create(b, loc, cmp, vr, vNew);
            // spike output (i8: 0 or 1)
            Value one8  = arith::ConstantOp::create(b, loc, outElem, b.getIntegerAttr(outElem, 1));
            Value zero8 = arith::ConstantOp::create(b, loc, outElem, b.getIntegerAttr(outElem, 0));
            Value spike = arith::SelectOp::create(b, loc, cmp, one8, zero8);
            linalg::YieldOp::create(b, loc, ValueRange{vFinal, spike});
          });
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
//  Pattern: snn.li → linalg.generic (float or quantized dynamics)
//      1-state (voltage), voltage output — no threshold or spike
//===----------------------------------------------------------------------===//
struct LowerLI : public OpRewritePattern<snn::LIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(snn::LIOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value voltage = op.getVoltage();
    Value output = op.getOutput();

    AffineMap id = rewriter.getDimIdentityMap();
    SmallVector<AffineMap> maps = {id, id, id};
    SmallVector<utils::IteratorType> iterTypes = {utils::IteratorType::parallel};

    if (isFloatMemRef(input)) {
      double decay = op.getDecayFloat().convertToDouble();
      Type fTy = cast<MemRefType>(input.getType()).getElementType();

      linalg::GenericOp::create(rewriter,
          loc, TypeRange{}, ValueRange{input},
          ValueRange{voltage, output}, maps, iterTypes,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value s = args[0], v = args[1];
            // voltage = decay * voltage + input
            Value dc      = arith::ConstantOp::create(b, loc, fTy, b.getFloatAttr(fTy, decay));
            Value vScaled = arith::MulFOp::create(b, loc, dc, v);
            Value vNew    = arith::AddFOp::create(b, loc, vScaled, s);
            linalg::YieldOp::create(b, loc, ValueRange{vNew, vNew});
          });
    } else {
      // Quantized path (Q12)
      int64_t dScale   = op.getDScale();
      int64_t decayInt = op.getDecayInt();
      Type i32 = rewriter.getI32Type();

      linalg::GenericOp::create(rewriter,
          loc, TypeRange{}, ValueRange{input},
          ValueRange{voltage, output}, maps, iterTypes,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value s = args[0], v = args[1];

            Value shiftVal = arith::ConstantOp::create(b,
                loc, i32, b.getI32IntegerAttr(dScale));

            // voltage = (decay * v) >> d_scale + input
            Value dc       = arith::ConstantOp::create(b, loc, i32, b.getI32IntegerAttr(decayInt));
            Value vProd    = arith::MulIOp::create(b, loc, dc, v);
            Value vShifted = arith::ShRSIOp::create(b, loc, vProd, shiftVal);
            Value vNew     = arith::AddIOp::create(b, loc, vShifted, s);
            linalg::YieldOp::create(b, loc, ValueRange{vNew, vNew});
          });
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
//  Pattern: snn.cubali → linalg.generic (float or quantized dynamics)
//      2-state (current+voltage), voltage output — no threshold or spike
//===----------------------------------------------------------------------===//
struct LowerCubaLI : public OpRewritePattern<snn::CubaLIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(snn::CubaLIOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value current = op.getCurrent();
    Value voltage = op.getVoltage();
    Value output = op.getOutput();

    AffineMap id = rewriter.getDimIdentityMap();
    SmallVector<AffineMap> maps = {id, id, id, id};
    SmallVector<utils::IteratorType> iterTypes = {utils::IteratorType::parallel};

    if (isFloatMemRef(input)) {
      double curDecay = op.getCurDecayFloat().convertToDouble();
      double volDecay = op.getVolDecayFloat().convertToDouble();
      Type fTy = cast<MemRefType>(input.getType()).getElementType();

      linalg::GenericOp::create(rewriter,
          loc, TypeRange{}, ValueRange{input},
          ValueRange{current, voltage, output}, maps, iterTypes,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value s = args[0], c = args[1], v = args[2];
            Value cd      = arith::ConstantOp::create(b, loc, fTy, b.getFloatAttr(fTy, curDecay));
            Value cScaled = arith::MulFOp::create(b, loc, cd, c);
            Value cNew    = arith::AddFOp::create(b, loc, cScaled, s);
            Value vd      = arith::ConstantOp::create(b, loc, fTy, b.getFloatAttr(fTy, volDecay));
            Value vScaled = arith::MulFOp::create(b, loc, vd, v);
            Value vNew    = arith::AddFOp::create(b, loc, vScaled, cNew);
            linalg::YieldOp::create(b, loc, ValueRange{cNew, vNew, vNew});
          });
    } else {
      // Quantized path (Q12)
      int64_t dScale      = op.getDScale();
      int64_t curDecayInt = op.getCurDecayInt();
      int64_t volDecayInt = op.getVolDecayInt();
      Type i32 = rewriter.getI32Type();

      linalg::GenericOp::create(rewriter,
          loc, TypeRange{}, ValueRange{input},
          ValueRange{current, voltage, output}, maps, iterTypes,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value s = args[0], c = args[1], v = args[2];

            Value shiftVal = arith::ConstantOp::create(b,
                loc, i32, b.getI32IntegerAttr(dScale));

            Value cd       = arith::ConstantOp::create(b, loc, i32, b.getI32IntegerAttr(curDecayInt));
            Value cProd    = arith::MulIOp::create(b, loc, cd, c);
            Value cShifted = arith::ShRSIOp::create(b, loc, cProd, shiftVal);
            Value cNew     = arith::AddIOp::create(b, loc, cShifted, s);

            Value vd       = arith::ConstantOp::create(b, loc, i32, b.getI32IntegerAttr(volDecayInt));
            Value vProd    = arith::MulIOp::create(b, loc, vd, v);
            Value vShifted = arith::ShRSIOp::create(b, loc, vProd, shiftVal);
            Value vNew     = arith::AddIOp::create(b, loc, vShifted, cNew);

            linalg::YieldOp::create(b, loc, ValueRange{cNew, vNew, vNew});
          });
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
//  Pass definition
//===----------------------------------------------------------------------===//
namespace snn {
#define GEN_PASS_DEF_CONVERTSNNTOLINALG
#include "SNN/Conversion/Passes.h.inc"
} // namespace snn

namespace {
struct ConvertSNNToLinalgPass
    : public snn::impl::ConvertSNNToLinalgBase<ConvertSNNToLinalgPass> {
  using ConvertSNNToLinalgBase::ConvertSNNToLinalgBase;

  // getArgument/getDescription/getDependentDialects come from the generated
  // base (declared in Passes.td).
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerLinear, LowerRescale, LowerCubaLIF, LowerCubaLI, LowerLIF, LowerLI>(&getContext());

    ConversionTarget target(getContext());
    // SNN ops must be eliminated
    target.addIllegalDialect<snn::SNNDialect>();
    // Everything else is legal
    target.addLegalDialect<linalg::LinalgDialect, arith::ArithDialect,
                           memref::MemRefDialect, func::FuncDialect>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> snn::createConvertSNNToLinalgPass() {
  return std::make_unique<ConvertSNNToLinalgPass>();
}

void snn::registerConvertSNNToLinalgPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createConvertSNNToLinalgPass();
  });
}
