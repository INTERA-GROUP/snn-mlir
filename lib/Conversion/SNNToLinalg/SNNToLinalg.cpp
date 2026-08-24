// Copyright 2026 N Vision Systems And Technologies SL
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
//  Helpers: the iteration space of a shape-preserving elementwise op
//
//  The neuron ops and snn.rescale read and write one element per element, so
//  their iteration space is just the operand shape: one identity map per
//  operand, one parallel iterator per dimension. Deriving both from the operand
//  rank is what lets a dense layer's vector and a conv layer's feature map share
//  a single lowering. At rank 1 these reproduce the previous
//  getDimIdentityMap() output exactly.
//===----------------------------------------------------------------------===//
static SmallVector<AffineMap> identityMaps(OpBuilder &b, Value operand,
                                           unsigned numOperands) {
  unsigned rank = cast<MemRefType>(operand.getType()).getRank();
  return SmallVector<AffineMap>(numOperands, b.getMultiDimIdentityMap(rank));
}

static SmallVector<utils::IteratorType> parallelIterators(Value operand) {
  unsigned rank = cast<MemRefType>(operand.getType()).getRank();
  return SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel);
}

//===----------------------------------------------------------------------===//
//  Helper: the Q12 decay step  (decay * state) >> d_scale
//
//  Both factors are Q12, so the product is Q24 and overflows i32 while the
//  state itself is still far from full. It is therefore computed in i64 and
//  truncated back, which puts the usable bound on the state width itself
//  (|state| < 2^31) instead of 2^31 >> d_scale. State memrefs and the kernel
//  ABI stay i32. Arithmetic width is a property of the dialect, not of the
//  target: host and RV32 must stay numerically identical.
//===----------------------------------------------------------------------===//
static Value decayMul(OpBuilder &b, Location loc, int64_t decayInt, Value state,
                      int64_t dScale) {
  Type i64 = b.getI64Type();
  Value dc   = arith::ConstantOp::create(b, loc, i64, b.getI64IntegerAttr(decayInt));
  Value st   = arith::ExtSIOp::create(b, loc, i64, state);
  Value prod = arith::MulIOp::create(b, loc, dc, st);
  Value sh   = arith::ConstantOp::create(b, loc, i64, b.getI64IntegerAttr(dScale));
  return arith::TruncIOp::create(b, loc, state.getType(),
                                 arith::ShRSIOp::create(b, loc, prod, sh));
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
//  Pattern: snn.conv2d → linalg.conv_2d_nchw_fchw (float)
//
//  Activations are rank-3 [C, H, W]; the named linalg conv is rank-4 (it wants
//  a leading batch axis), so the pattern brackets the conv with a unit-batch
//  memref.expand_shape on both input and output. Padding is materialized
//  explicitly — the named conv models only the valid convolution, so a padded
//  input is a fresh zero-filled buffer with the real input copied into its
//  interior (the bufferized form of tensor.pad). Bias, when present, is a
//  per-output-channel value broadcast over the spatial dimensions.
//===----------------------------------------------------------------------===//
struct LowerConv2d : public OpRewritePattern<snn::Conv2dOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(snn::Conv2dOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value weights = op.getWeights();
    Value output = op.getOutput();

    if (!isFloatMemRef(input))
      return rewriter.notifyMatchFailure(
          op, "only the float conv2d lowering is implemented");

    auto inTy = cast<MemRefType>(input.getType());
    auto outTy = cast<MemRefType>(output.getType());
    Type elemTy = inTy.getElementType();
    int64_t C = inTy.getDimSize(0), H = inTy.getDimSize(1), W = inTy.getDimSize(2);

    ArrayRef<int64_t> stride = op.getStride();
    ArrayRef<int64_t> padding = op.getPadding();
    int64_t ph = padding[0], pw = padding[1];

    // ── materialize padding (valid-conv only), else convolve the input ────────
    Value convInput = input;
    if (ph != 0 || pw != 0) {
      int64_t Hp = H + 2 * ph, Wp = W + 2 * pw;
      auto padTy = MemRefType::get({C, Hp, Wp}, elemTy);
      Value padded = memref::AllocaOp::create(rewriter, loc, padTy);
      Value fzero = arith::ConstantOp::create(rewriter, loc, elemTy,
                                              rewriter.getFloatAttr(elemTy, 0.0));
      linalg::FillOp::create(rewriter, loc, fzero, padded);
      // Copy the real input into the [0:C, ph:ph+H, pw:pw+W] interior.
      Value interior = memref::SubViewOp::create(
          rewriter, loc, padded,
          /*offsets=*/ArrayRef<int64_t>{0, ph, pw},
          /*sizes=*/ArrayRef<int64_t>{C, H, W},
          /*strides=*/ArrayRef<int64_t>{1, 1, 1});
      memref::CopyOp::create(rewriter, loc, input, interior);
      convInput = padded;
    }

    // ── bracket with the unit batch dimension the named conv wants ────────────
    // reassociation [[0,1],[2],[3]]: batch is folded onto the channel axis.
    SmallVector<ReassociationIndices> reassoc = {{0, 1}, {2}, {3}};
    auto conv4dTy = [&](Value v) {
      auto t = cast<MemRefType>(v.getType());
      return MemRefType::get({1, t.getDimSize(0), t.getDimSize(1),
                              t.getDimSize(2)},
                             elemTy);
    };
    Value in4d = memref::ExpandShapeOp::create(rewriter, loc, conv4dTy(convInput),
                                               convInput, reassoc);
    Value out4d = memref::ExpandShapeOp::create(rewriter, loc, conv4dTy(output),
                                                output, reassoc);

    // conv_2d_nchw_fchw accumulates into its output, so zero it first.
    Value fzero = arith::ConstantOp::create(rewriter, loc, elemTy,
                                            rewriter.getFloatAttr(elemTy, 0.0));
    linalg::FillOp::create(rewriter, loc, fzero, out4d);

    auto i64Vec = [&](ArrayRef<int64_t> v) {
      return rewriter.getI64TensorAttr(v);
    };
    linalg::Conv2DNchwFchwOp::create(
        rewriter, loc, TypeRange{}, ValueRange{in4d, weights},
        ValueRange{out4d},
        /*strides=*/i64Vec(stride), /*dilations=*/i64Vec({1, 1}));

    // ── optional per-output-channel bias, broadcast over H×W ──────────────────
    Value bias = op.getBias();
    if (bias) {
      MLIRContext *ctx = rewriter.getContext();
      SmallVector<AffineMap> biasMaps = {
          AffineMap::get(3, 0, {rewriter.getAffineDimExpr(0)}, ctx), // bias[o]
          rewriter.getMultiDimIdentityMap(3),                        // out[o,h,w]
      };
      SmallVector<utils::IteratorType> biasIter(3, utils::IteratorType::parallel);
      linalg::GenericOp::create(
          rewriter, loc, TypeRange{}, ValueRange{bias}, ValueRange{output},
          biasMaps, biasIter,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value bval = args[0], acc = args[1];
            Value sum = arith::AddFOp::create(b, loc, acc, bval);
            linalg::YieldOp::create(b, loc, sum);
          });
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

    SmallVector<AffineMap> maps = identityMaps(rewriter, input, 2);
    SmallVector<utils::IteratorType> iterTypes = parallelIterators(input);

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

    SmallVector<AffineMap> maps = identityMaps(rewriter, input, 4);
    SmallVector<utils::IteratorType> iterTypes = parallelIterators(input);

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

      // State stays i32; only the decay product widens (see decayMul).
      Type i32 = rewriter.getI32Type();
      auto outElem =
          cast<MemRefType>(output.getType()).getElementType(); // i8

      linalg::GenericOp::create(rewriter,
          loc, TypeRange{}, ValueRange{input},
          ValueRange{current, voltage, output}, maps, iterTypes,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value s = args[0], c = args[1], v = args[2];

            // c_new = (cur_decay * c) >> d_scale + input
            Value cNew = arith::AddIOp::create(b, loc,
                decayMul(b, loc, curDecayInt, c, dScale), s);

            // v_new = (vol_decay * v) >> d_scale + c_new
            Value vNew = arith::AddIOp::create(b, loc,
                decayMul(b, loc, volDecayInt, v, dScale), cNew);
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

    SmallVector<AffineMap> maps = identityMaps(rewriter, input, 3);
    SmallVector<utils::IteratorType> iterTypes = parallelIterators(input);

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

            // voltage = (decay * v) >> d_scale + input
            Value vNew = arith::AddIOp::create(b, loc,
                decayMul(b, loc, decayInt, v, dScale), s);
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

    SmallVector<AffineMap> maps = identityMaps(rewriter, input, 3);
    SmallVector<utils::IteratorType> iterTypes = parallelIterators(input);

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

      linalg::GenericOp::create(rewriter,
          loc, TypeRange{}, ValueRange{input},
          ValueRange{voltage, output}, maps, iterTypes,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value s = args[0], v = args[1];

            // voltage = (decay * v) >> d_scale + input
            Value vNew = arith::AddIOp::create(b, loc,
                decayMul(b, loc, decayInt, v, dScale), s);
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

    SmallVector<AffineMap> maps = identityMaps(rewriter, input, 4);
    SmallVector<utils::IteratorType> iterTypes = parallelIterators(input);

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

      linalg::GenericOp::create(rewriter,
          loc, TypeRange{}, ValueRange{input},
          ValueRange{current, voltage, output}, maps, iterTypes,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value s = args[0], c = args[1], v = args[2];

            Value cNew = arith::AddIOp::create(b, loc,
                decayMul(b, loc, curDecayInt, c, dScale), s);

            Value vNew = arith::AddIOp::create(b, loc,
                decayMul(b, loc, volDecayInt, v, dScale), cNew);

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
    patterns.add<LowerLinear, LowerConv2d, LowerRescale, LowerCubaLIF, LowerCubaLI, LowerLIF, LowerLI>(&getContext());

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
