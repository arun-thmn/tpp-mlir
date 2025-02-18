//===- BrgemmLinalgTiling.cpp--------------------------------------*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements parallel loop insertion for tiling.
//
//===----------------------------------------------------------------------===//
#include "TPP/Transforms/Transforms.h"
#include "TPP/Transforms/Utils/VNNIUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "brgemm-linalg-tiling"

namespace mlir {
namespace tpp {
#define GEN_PASS_DECL_BRGEMMLINALGTILING
#define GEN_PASS_DEF_BRGEMMLINALGTILING
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;
using namespace mlir::tpp;

namespace mlir {
namespace tpp {

template <typename BrgemmOp>
struct LinalgOpTiling : OpRewritePattern<BrgemmOp> {
  using OpRewritePattern<BrgemmOp>::OpRewritePattern;

  LinalgOpTiling(MLIRContext *ctx, BrgemmLinalgTilingOptions tilingoptions)
      : OpRewritePattern<BrgemmOp>(ctx), options(tilingoptions) {}

  LogicalResult matchAndRewrite(BrgemmOp brgemmOp,
                                PatternRewriter &rewriter) const override {
    if (!brgemmOp.hasPureBufferSemantics())
      return failure();

    // Check whether the tile sizes are valid
    if (options.registerTileShape.size() != 3)
      return rewriter.notifyMatchFailure(
          brgemmOp, "Invalid user input tile sizes. Should be <m,n,k>");

    // Check the whether the operation is brmatmul fp32 or bf16 type using
    // reduction count
    SmallVector<utils::IteratorType> brgemmIteratorTypes =
        brgemmOp.getIteratorTypesArray();
    int reductionCount =
        std::count(brgemmIteratorTypes.begin(), brgemmIteratorTypes.end(),
                   utils::IteratorType::reduction);

    if (reductionCount == 0)
      return rewriter.notifyMatchFailure(brgemmOp,
                                         "Matmul operation not supported yet");

    if (reductionCount == 1)
      return rewriter.notifyMatchFailure(
          brgemmOp, "Batch matmul operation not supported yet");

    if (reductionCount > 3)
      return rewriter.notifyMatchFailure(brgemmOp,
                                         "The operation is not a gemm");

    auto tensorShapeLhs =
        dyn_cast<MemRefType>(brgemmOp.getOperand(0).getType()).getShape();
    auto tensorShapeRhs =
        dyn_cast<MemRefType>(brgemmOp.getOperand(1).getType()).getShape();

    if (reductionCount == 2 &&
        (tensorShapeLhs.size() != 3 || tensorShapeRhs.size() != 3))
      return rewriter.notifyMatchFailure(
          brgemmOp, "Invalid rank for batch reduce operation");

    auto vnniOpt = vnni::utils::isInVnniLayout(brgemmOp);
    if (reductionCount == 3 && !vnniOpt)
      return rewriter.notifyMatchFailure(
          brgemmOp,
          "Failed matching for batch reduce operation with vnni layout");

    //  Get the register blocking tile shape from the user input
    SmallVector<int64_t> mxnxkTile(options.registerTileShape.begin(),
                                   options.registerTileShape.end());

    linalg::LinalgTilingOptions options;
    options.setLoopType(linalg::LinalgTilingLoopType::Loops);
    FailureOr<linalg::TiledLinalgOp> tiledOp;

    if (vnniOpt) {
      // k-tile size adjusted based on the vnni layout for bf16 type
      auto tensorShape =
          dyn_cast<MemRefType>(brgemmOp.getOperand(0).getType()).getShape();
      auto kTileVnni = mxnxkTile[2] / tensorShape[3];

      // Note: We make an assumption that the k tile size is divisible to 
      // the powers of 2.
      if (kTileVnni < 1)
	  return rewriter.notifyMatchFailure(
            brgemmOp, "Failed matching K tile size for batch reduce operation "
                      "with vnni layout. K tile size should be >= vnni layout");

      mxnxkTile[2] = kTileVnni;
      // Tile options for bf16 type with vnni layout
      options.setTileSizes({1, 0, mxnxkTile[0], mxnxkTile[1], mxnxkTile[2]});
      options.setInterchange({2, 3, 0, 4, 1});
      tiledOp =
          linalg::tileLinalgOp(rewriter, brgemmOp, options);

    } else {
      // Tile options for f32 type. 
      options.setTileSizes({1, mxnxkTile[0], mxnxkTile[1], mxnxkTile[2]});
      options.setInterchange({1, 2, 0, 3});
      tiledOp =
          linalg::tileLinalgOp(rewriter, brgemmOp, options);
    }

    if (failed(tiledOp)) {
        return failure();
      }
    rewriter.replaceOp(brgemmOp, tiledOp->op->getResults());

    return success();
  }

private:
  BrgemmLinalgTilingOptions options;
};

void populateBrgemmLinalgTilingPatterns(RewritePatternSet &patterns,
                                        BrgemmLinalgTilingOptions options) {
  patterns.add<LinalgOpTiling<linalg::GenericOp>,
               LinalgOpTiling<linalg::BatchReduceMatmulOp>>(
      patterns.getContext(), options);
}

struct BrgemmLinalgTiling
    : public tpp::impl::BrgemmLinalgTilingBase<BrgemmLinalgTiling> {

  using BrgemmLinalgTilingBase::BrgemmLinalgTilingBase;

  void runOnOperation() override {
    BrgemmLinalgTilingOptions options;
    options.registerTileShape = SmallVector<unsigned>{*registerTileShape};
    RewritePatternSet patterns(&getContext());
    populateBrgemmLinalgTilingPatterns(patterns, options);
    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;

    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                       config);
  }
};
} // namespace tpp
} // namespace mlir
