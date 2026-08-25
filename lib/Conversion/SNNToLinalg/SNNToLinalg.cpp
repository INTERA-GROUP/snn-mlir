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
//  Helper: a typed zero constant (float or integer element type)
//===----------------------------------------------------------------------===//
static Value zeroOf(OpBuilder &b, Location loc, Type elem) {
  if (isa<FloatType>(elem))
    return arith::ConstantOp::create(b, loc, elem, b.getFloatAttr(elem, 0.0));
  return arith::ConstantOp::create(b, loc, elem, b.getIntegerAttr(elem, 0));
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

    auto inTy = cast<MemRefType>(input.getType());
    auto outTy = cast<MemRefType>(output.getType());
    Type inElem = inTy.getElementType();   // f32 or i8
    Type outElem = outTy.getElementType(); // f32 or i32
    bool quant = !isFloatMemRef(input);
    int64_t C = inTy.getDimSize(0), H = inTy.getDimSize(1), W = inTy.getDimSize(2);

    ArrayRef<int64_t> stride = op.getStride();
    ArrayRef<int64_t> padding = op.getPadding();
    int64_t ph = padding[0], pw = padding[1];

    // ── materialize padding (valid-conv only), else convolve the input ────────
    // Padding with 0 is correct in both modes: 0 is the symmetric zero-point.
    Value convInput = input;
    if (ph != 0 || pw != 0) {
      int64_t Hp = H + 2 * ph, Wp = W + 2 * pw;
      auto padTy = MemRefType::get({C, Hp, Wp}, inElem);
      Value padded = memref::AllocaOp::create(rewriter, loc, padTy);
      linalg::FillOp::create(rewriter, loc, zeroOf(rewriter, loc, inElem), padded);
      // Copy the real input into the [0:C, ph:ph+H, pw:pw+W] interior.
      Value interior = memref::SubViewOp::create(
          rewriter, loc, padded,
          /*offsets=*/ArrayRef<int64_t>{0, ph, pw},
          /*sizes=*/ArrayRef<int64_t>{C, H, W},
          /*strides=*/ArrayRef<int64_t>{1, 1, 1});
      linalg::CopyOp::create(rewriter, loc, input, interior);
      convInput = padded;
    }

    // ── bracket with the unit batch dimension the named conv wants ────────────
    // reassociation [[0,1],[2],[3]]: batch is folded onto the channel axis.
    SmallVector<ReassociationIndices> reassoc = {{0, 1}, {2}, {3}};
    auto conv4dTy = [&](Value v) {
      auto t = cast<MemRefType>(v.getType());
      return MemRefType::get({1, t.getDimSize(0), t.getDimSize(1),
                              t.getDimSize(2)},
                             t.getElementType());
    };
    Value in4d = memref::ExpandShapeOp::create(rewriter, loc, conv4dTy(convInput),
                                               convInput, reassoc);
    Value out4d = memref::ExpandShapeOp::create(rewriter, loc, conv4dTy(output),
                                                output, reassoc);

    // The named conv accumulates into its output, so zero it first.
    linalg::FillOp::create(rewriter, loc, zeroOf(rewriter, loc, outElem), out4d);

    auto i64Vec = [&](ArrayRef<int64_t> v) {
      return rewriter.getI64TensorAttr(v);
    };
    if (quant) {
      // Symmetric quantization: both zero-points are 0.
      Value izp = zeroOf(rewriter, loc, rewriter.getI32Type());
      Value kzp = zeroOf(rewriter, loc, rewriter.getI32Type());
      linalg::Conv2DNchwFchwQOp::create(
          rewriter, loc, TypeRange{}, ValueRange{in4d, weights, izp, kzp},
          ValueRange{out4d},
          /*strides=*/i64Vec(stride), /*dilations=*/i64Vec({1, 1}));
    } else {
      linalg::Conv2DNchwFchwOp::create(
          rewriter, loc, TypeRange{}, ValueRange{in4d, weights},
          ValueRange{out4d},
          /*strides=*/i64Vec(stride), /*dilations=*/i64Vec({1, 1}));
    }

    // ── optional per-output-channel bias, broadcast over H×W ──────────────────
    // Quantized bias is i32 (same scale as the MAC accumulator).
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
            Value sum = quant ? arith::AddIOp::create(b, loc, acc, bval).getResult()
                              : arith::AddFOp::create(b, loc, acc, bval).getResult();
            linalg::YieldOp::create(b, loc, sum);
          });
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
//  Pattern: snn.conv1d → linalg.conv_1d_ncw_fcw (float)
//                      → linalg.conv_2d_nchw_fchw_q (quantized, 2-D embed)
//
//  The 1-D analogue of LowerConv2d: padding is materialized explicitly
//  (valid-conv only) and an optional per-output-channel bias is broadcast over
//  the spatial dimension. The float path brackets the rank-2 [C, L] activations
//  with a unit-batch memref.expand_shape so the named rank-3 conv_1d applies.
//  There is no quantized rank-3 named conv, so the quantized path instead embeds
//  the 1-D convolution in a 2-D one with a unit width axis — [C, L] → [1,C,L,1],
//  weights [O,C,K] → [O,C,K,1] — and reuses conv_2d_nchw_fchw_q with a unit W
//  kernel and stride. The [1,O,Lo,1] result aliases the rank-2 output buffer.
//===----------------------------------------------------------------------===//
struct LowerConv1d : public OpRewritePattern<snn::Conv1dOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(snn::Conv1dOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value weights = op.getWeights();
    Value output = op.getOutput();

    auto inTy = cast<MemRefType>(input.getType());
    auto outTy = cast<MemRefType>(output.getType());
    Type inElem = inTy.getElementType();   // f32 or i8
    Type outElem = outTy.getElementType(); // f32 or i32
    bool quant = !isFloatMemRef(input);
    int64_t C = inTy.getDimSize(0), L = inTy.getDimSize(1);

    int64_t stride = op.getStride();
    int64_t pad = op.getPadding();

    // ── materialize padding (valid-conv only), else convolve the input ────────
    // Padding with 0 is correct in both modes: 0 is the symmetric zero-point.
    Value convInput = input;
    if (pad != 0) {
      int64_t Lp = L + 2 * pad;
      auto padTy = MemRefType::get({C, Lp}, inElem);
      Value padded = memref::AllocaOp::create(rewriter, loc, padTy);
      linalg::FillOp::create(rewriter, loc, zeroOf(rewriter, loc, inElem), padded);
      // Copy the real input into the [0:C, pad:pad+L] interior.
      Value interior = memref::SubViewOp::create(
          rewriter, loc, padded,
          /*offsets=*/ArrayRef<int64_t>{0, pad},
          /*sizes=*/ArrayRef<int64_t>{C, L},
          /*strides=*/ArrayRef<int64_t>{1, 1});
      linalg::CopyOp::create(rewriter, loc, input, interior);
      convInput = padded;
    }

    auto i64Vec = [&](ArrayRef<int64_t> v) {
      return rewriter.getI64TensorAttr(v);
    };

    if (quant) {
      // Embed in 2-D with a trailing unit width axis (see the header note).
      // reassociation [[0,1],[2,3]] on activations: [1,C] and [L,1] fold onto
      // the source [C, L]; [[0],[1],[2,3]] on weights adds the unit Kw.
      SmallVector<ReassociationIndices> actReassoc = {{0, 1}, {2, 3}};
      SmallVector<ReassociationIndices> wReassoc = {{0}, {1}, {2, 3}};
      auto ct = cast<MemRefType>(convInput.getType());
      auto embedTy = [&](Value v, ArrayRef<int64_t> shape) {
        return MemRefType::get(shape,
                               cast<MemRefType>(v.getType()).getElementType());
      };
      Value in4d = memref::ExpandShapeOp::create(
          rewriter, loc,
          embedTy(convInput, {1, ct.getDimSize(0), ct.getDimSize(1), 1}),
          convInput, actReassoc);
      auto wt = cast<MemRefType>(weights.getType());
      Value w4d = memref::ExpandShapeOp::create(
          rewriter, loc,
          embedTy(weights, {wt.getDimSize(0), wt.getDimSize(1), wt.getDimSize(2),
                            1}),
          weights, wReassoc);
      Value out4d = memref::ExpandShapeOp::create(
          rewriter, loc,
          embedTy(output, {1, outTy.getDimSize(0), outTy.getDimSize(1), 1}),
          output, actReassoc);

      // The named conv accumulates into its output, so zero it first.
      linalg::FillOp::create(rewriter, loc, zeroOf(rewriter, loc, outElem), out4d);
      // Symmetric quantization: both zero-points are 0.
      Value izp = zeroOf(rewriter, loc, rewriter.getI32Type());
      Value kzp = zeroOf(rewriter, loc, rewriter.getI32Type());
      linalg::Conv2DNchwFchwQOp::create(
          rewriter, loc, TypeRange{}, ValueRange{in4d, w4d, izp, kzp},
          ValueRange{out4d},
          /*strides=*/i64Vec({stride, 1}), /*dilations=*/i64Vec({1, 1}));
    } else {
      // ── bracket with the unit batch dimension the named conv wants ──────────
      // reassociation [[0,1],[2]]: batch is folded onto the channel axis.
      SmallVector<ReassociationIndices> reassoc = {{0, 1}, {2}};
      auto conv3dTy = [&](Value v) {
        auto t = cast<MemRefType>(v.getType());
        return MemRefType::get({1, t.getDimSize(0), t.getDimSize(1)},
                               t.getElementType());
      };
      Value in3d = memref::ExpandShapeOp::create(
          rewriter, loc, conv3dTy(convInput), convInput, reassoc);
      Value out3d = memref::ExpandShapeOp::create(rewriter, loc, conv3dTy(output),
                                                  output, reassoc);

      // conv_1d_ncw_fcw accumulates into its output, so zero it first.
      linalg::FillOp::create(rewriter, loc, zeroOf(rewriter, loc, outElem), out3d);
      linalg::Conv1DNcwFcwOp::create(
          rewriter, loc, TypeRange{}, ValueRange{in3d, weights},
          ValueRange{out3d},
          /*strides=*/i64Vec({stride}), /*dilations=*/i64Vec({1}));
    }

    // ── optional per-output-channel bias, broadcast over L ────────────────────
    // Quantized bias is i32 (same scale as the MAC accumulator).
    Value bias = op.getBias();
    if (bias) {
      MLIRContext *ctx = rewriter.getContext();
      SmallVector<AffineMap> biasMaps = {
          AffineMap::get(2, 0, {rewriter.getAffineDimExpr(0)}, ctx), // bias[o]
          rewriter.getMultiDimIdentityMap(2),                        // out[o,l]
      };
      SmallVector<utils::IteratorType> biasIter(2, utils::IteratorType::parallel);
      linalg::GenericOp::create(
          rewriter, loc, TypeRange{}, ValueRange{bias}, ValueRange{output},
          biasMaps, biasIter,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value bval = args[0], acc = args[1];
            Value sum = quant ? arith::AddIOp::create(b, loc, acc, bval).getResult()
                              : arith::AddFOp::create(b, loc, acc, bval).getResult();
            linalg::YieldOp::create(b, loc, sum);
          });
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
//  Pattern: snn.sumpool2d → linalg.pooling_nchw_sum (float)
//
//  Same bracketing as LowerConv2d, minus the weights: rank-3 [C, H, W]
//  activations are padded explicitly (valid-pool only), expanded with a
//  unit-batch memref.expand_shape to the [N, C, H, W] the named pooling op
//  wants, and reduced by linalg.pooling_nchw_sum. That op takes a window-shaped
//  operand whose *values are ignored* — only its shape [Kh, Kw] defines the
//  window — so a throwaway alloca stands in for it.
//===----------------------------------------------------------------------===//
struct LowerSumPool2d : public OpRewritePattern<snn::SumPool2dOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(snn::SumPool2dOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value output = op.getOutput();

    auto inTy = cast<MemRefType>(input.getType());
    Type elemTy = inTy.getElementType(); // f32 or i8 (scale-preserving: i8 -> i8)
    int64_t C = inTy.getDimSize(0), H = inTy.getDimSize(1), W = inTy.getDimSize(2);

    ArrayRef<int64_t> kernel = op.getKernel();
    ArrayRef<int64_t> stride = op.getStride();
    ArrayRef<int64_t> padding = op.getPadding();
    int64_t ph = padding[0], pw = padding[1];

    // ── materialize padding (valid-pool only), else pool the input ────────────
    Value poolInput = input;
    if (ph != 0 || pw != 0) {
      int64_t Hp = H + 2 * ph, Wp = W + 2 * pw;
      auto padTy = MemRefType::get({C, Hp, Wp}, elemTy);
      Value padded = memref::AllocaOp::create(rewriter, loc, padTy);
      linalg::FillOp::create(rewriter, loc, zeroOf(rewriter, loc, elemTy), padded);
      Value interior = memref::SubViewOp::create(
          rewriter, loc, padded,
          /*offsets=*/ArrayRef<int64_t>{0, ph, pw},
          /*sizes=*/ArrayRef<int64_t>{C, H, W},
          /*strides=*/ArrayRef<int64_t>{1, 1, 1});
      linalg::CopyOp::create(rewriter, loc, input, interior);
      poolInput = padded;
    }

    // ── bracket with the unit batch dimension the named pooling op wants ──────
    // reassociation [[0,1],[2],[3]]: batch is folded onto the channel axis.
    SmallVector<ReassociationIndices> reassoc = {{0, 1}, {2}, {3}};
    auto pool4dTy = [&](Value v) {
      auto t = cast<MemRefType>(v.getType());
      return MemRefType::get({1, t.getDimSize(0), t.getDimSize(1),
                              t.getDimSize(2)},
                             elemTy);
    };
    Value in4d = memref::ExpandShapeOp::create(rewriter, loc, pool4dTy(poolInput),
                                               poolInput, reassoc);
    Value out4d = memref::ExpandShapeOp::create(rewriter, loc, pool4dTy(output),
                                                output, reassoc);

    // pooling_nchw_sum accumulates into its output, so zero it first.
    linalg::FillOp::create(rewriter, loc, zeroOf(rewriter, loc, elemTy), out4d);

    // The window operand: only its shape [Kh, Kw] matters to the named op.
    auto winTy = MemRefType::get({kernel[0], kernel[1]}, elemTy);
    Value window = memref::AllocaOp::create(rewriter, loc, winTy);

    auto i64Vec = [&](ArrayRef<int64_t> v) {
      return rewriter.getI64TensorAttr(v);
    };
    linalg::PoolingNchwSumOp::create(
        rewriter, loc, TypeRange{}, ValueRange{in4d, window},
        ValueRange{out4d},
        /*strides=*/i64Vec(stride), /*dilations=*/i64Vec({1, 1}));

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
//  Pattern: snn.avgpool2d → linalg.pooling_nchw_sum + divide (float)
//
//  The sum-pool lowering, then an in-place divide of every output element by the
//  window count (kh*kw). There is no named pooling_nchw_avg in this LLVM.
//===----------------------------------------------------------------------===//
struct LowerAvgPool2d : public OpRewritePattern<snn::AvgPool2dOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(snn::AvgPool2dOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value output = op.getOutput();

    auto inTy = cast<MemRefType>(input.getType());
    Type elemTy = inTy.getElementType(); // f32 or i8 (truncating integer mean)
    bool quant = !isFloatMemRef(input);
    int64_t C = inTy.getDimSize(0), H = inTy.getDimSize(1), W = inTy.getDimSize(2);

    ArrayRef<int64_t> kernel = op.getKernel();
    ArrayRef<int64_t> stride = op.getStride();
    ArrayRef<int64_t> padding = op.getPadding();
    int64_t ph = padding[0], pw = padding[1];

    // ── materialize padding (valid-pool only), else pool the input ────────────
    Value poolInput = input;
    if (ph != 0 || pw != 0) {
      int64_t Hp = H + 2 * ph, Wp = W + 2 * pw;
      auto padTy = MemRefType::get({C, Hp, Wp}, elemTy);
      Value padded = memref::AllocaOp::create(rewriter, loc, padTy);
      linalg::FillOp::create(rewriter, loc, zeroOf(rewriter, loc, elemTy), padded);
      Value interior = memref::SubViewOp::create(
          rewriter, loc, padded,
          /*offsets=*/ArrayRef<int64_t>{0, ph, pw},
          /*sizes=*/ArrayRef<int64_t>{C, H, W},
          /*strides=*/ArrayRef<int64_t>{1, 1, 1});
      linalg::CopyOp::create(rewriter, loc, input, interior);
      poolInput = padded;
    }

    // ── bracket with the unit batch dimension the named pooling op wants ──────
    SmallVector<ReassociationIndices> reassoc = {{0, 1}, {2}, {3}};
    auto pool4dTy = [&](Value v) {
      auto t = cast<MemRefType>(v.getType());
      return MemRefType::get({1, t.getDimSize(0), t.getDimSize(1),
                              t.getDimSize(2)},
                             elemTy);
    };
    Value in4d = memref::ExpandShapeOp::create(rewriter, loc, pool4dTy(poolInput),
                                               poolInput, reassoc);
    Value out4d = memref::ExpandShapeOp::create(rewriter, loc, pool4dTy(output),
                                                output, reassoc);

    // pooling_nchw_sum accumulates into its output, so zero it first.
    linalg::FillOp::create(rewriter, loc, zeroOf(rewriter, loc, elemTy), out4d);

    auto winTy = MemRefType::get({kernel[0], kernel[1]}, elemTy);
    Value window = memref::AllocaOp::create(rewriter, loc, winTy);

    auto i64Vec = [&](ArrayRef<int64_t> v) {
      return rewriter.getI64TensorAttr(v);
    };
    linalg::PoolingNchwSumOp::create(
        rewriter, loc, TypeRange{}, ValueRange{in4d, window},
        ValueRange{out4d},
        /*strides=*/i64Vec(stride), /*dilations=*/i64Vec({1, 1}));

    // ── divide each summed window by its count → the mean ─────────────────────
    // Float: exact division; quantized: truncating signed integer division
    // (count-include-pad, so the divisor is the full window area).
    int64_t count = kernel[0] * kernel[1];
    Value divisor = quant
        ? arith::ConstantOp::create(rewriter, loc, elemTy,
                                    rewriter.getIntegerAttr(elemTy, count))
        : arith::ConstantOp::create(rewriter, loc, elemTy,
                                    rewriter.getFloatAttr(elemTy, double(count)));
    SmallVector<AffineMap> divMaps = {rewriter.getMultiDimIdentityMap(3)};
    SmallVector<utils::IteratorType> divIter(3, utils::IteratorType::parallel);
    linalg::GenericOp::create(
        rewriter, loc, TypeRange{}, ValueRange{}, ValueRange{output},
        divMaps, divIter,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value q = quant ? arith::DivSIOp::create(b, loc, args[0], divisor).getResult()
                          : arith::DivFOp::create(b, loc, args[0], divisor).getResult();
          linalg::YieldOp::create(b, loc, q);
        });

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
    patterns.add<LowerLinear, LowerConv2d, LowerConv1d, LowerSumPool2d, LowerAvgPool2d, LowerRescale, LowerCubaLIF, LowerCubaLI, LowerLIF, LowerLI>(&getContext());

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
