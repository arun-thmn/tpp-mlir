//===- VectorTiling.cpp -----------------------------------------*- C++-*-===//
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
#include "TPP/IR/TilingUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <iostream>
#include <list>
#include <set>
#define DEBUG_TYPE "linalg-tiling"

namespace mlir {
namespace tpp {
#define GEN_PASS_DECL_LINALGTILING
#define GEN_PASS_DEF_LINALGTILING
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;
using namespace mlir::tpp;
using namespace std;

namespace mlir {
namespace tpp {

namespace {
auto par = utils::IteratorType::parallel;
auto red = utils::IteratorType::reduction;
} // namespace

static llvm::SmallDenseSet<int64_t>
findPermutationsIndexingOperand(AffineMap indexingMap,
                                ArrayRef<utils::IteratorType> iterators,
                                utils::IteratorType iter) {
  assert(iterators.size() == indexingMap.getNumDims());
  llvm::SmallDenseSet<int64_t> res;
  for (AffineExpr e : indexingMap.getResults()) {
    if (auto d = dyn_cast<AffineDimExpr>(e)) {
      if (iterators[d.getPosition()] == iter &&
          llvm::count_if(indexingMap.getResults(), [d](AffineExpr e) {
            return e.isFunctionOfDim(d.getPosition());
          }) == 1)
        res.insert(d.getPosition());
    }
  }
  return res;
}

//	namespace {

// template <typename LinalgOp>

struct LinalgOpTiling : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LinalgOpTiling(MLIRContext *ctx, LinalgTilingOptions tilingoptions)
      : OpRewritePattern(ctx), options(tilingoptions) {}

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {

    static list<linalg::GenericOp> visited;
    if (std::find(visited.begin(), visited.end(), linalgOp) != visited.end())
      return failure();
    visited.push_back(linalgOp);
    std::vector<int64_t> tileShapeM(options.mTileShape.begin(),
                                    options.mTileShape.end());
    std::vector<int64_t> tileShapeN(options.nTileShape.begin(),
                                    options.nTileShape.end());
    std::vector<int64_t> finaltile(3);

    std::set_union(tileShapeM.begin(), tileShapeM.end(), tileShapeN.begin(),
                   tileShapeN.end(), finaltile.begin());
    SmallVector<int64_t> boundariesOne{1,
                                       static_cast<long>(tileShapeM.size() - 1),
                                       static_cast<long>(finaltile.size() - 1)};

    rewriter.setInsertionPointToStart(linalgOp->getParentOp()->getBlock());
    int i = 0;
    SmallVector<SmallVector<Value>> inductionVars(finaltile.size());
    for (int i = 0; i < finaltile.size(); i++)
      inductionVars[i].reserve(finaltile.size());
    for (auto itrShapeM = finaltile.begin(); itrShapeM != finaltile.end();
         itrShapeM++, i++) {
      int index = i / boundariesOne[i];
      int offset = i / (finaltile.size() - 1);
      auto upperBound =
          dyn_cast<MemRefType>(linalgOp.getOperand(index).getType())
              .getShape()[offset];
      Location loc = linalgOp.getLoc();
      Value zeroCst = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value ubCst = rewriter.create<arith::ConstantIndexOp>(loc, upperBound);
      Value stepCst = rewriter.create<arith::ConstantIndexOp>(loc, *itrShapeM);
      scf::ForOp loopOp = rewriter.create<scf::ForOp>(linalgOp.getLoc(),
                                                      zeroCst, ubCst, stepCst);
      rewriter.setInsertionPointToStart(loopOp.getBody());

      inductionVars[index][offset] = inductionVars[offset][index] =
          loopOp.getInductionVar();
    }

    SmallVector<std::vector<int64_t>> tiles = {tileShapeM, tileShapeN};

    rewriter.setInsertionPointToStart(linalgOp.getBody());
    auto contractionDim = inferContractionDims(linalgOp);
    auto reductionDim = contractionDim->k.front();
    SmallVector<std::vector<int64_t>> tileshapes{tileShapeM, tileShapeN,
                                                 finaltile};
    for (size_t i = 0; i < linalgOp.getNumOperands(); i++) {
      SmallVector<int64_t> indecies;

      auto input = linalgOp.getOperand(i);
      auto operandType = input.getType();
      SmallVector<int64_t> shape;
      size_t index = 0;
      SmallVector<OpFoldResult> offsets;

      for (size_t j = 0;
           j < dyn_cast<ShapedType>(operandType).getShape().size(); j++) {
        if (dyn_cast<ShapedType>(operandType).getShape()[j] == reductionDim) {
          indecies.push_back(1);
        }

        for (size_t k = 0; k < tiles[i].size(); k++) {
          indecies.push_back(dyn_cast<ShapedType>(operandType).getShape()[j] /
                             tiles[i][k]);
          auto index = dyn_cast<ShapedType>(operandType).getShape().size() - j;
          offsets.push_back(inductionVars[index][k]);
          break;
        }

        auto subviewType = MemRefType::get(
            {indecies}, dyn_cast<MemRefType>(operandType).getElementType());

        auto [staticStrides, staticOffset] = getStridesAndOffset(subviewType);

        auto newSubviewType = MemRefType::get(
            {indecies}, dyn_cast<MemRefType>(operandType).getElementType(),
            StridedLayoutAttr::get(rewriter.getContext(), ShapedType::kDynamic,
                                   staticStrides));

        SmallVector<OpFoldResult> strides(indecies.size(),
                                          rewriter.getIndexAttr(1));

        SmallVector<OpFoldResult> shape;
        for (size_t k = 0; k < tileshapes[i].size(); k++)
          shape.push_back(rewriter.getIndexAttr(tileshapes[i][k]));
        auto subview = rewriter.create<memref::SubViewOp>(
            linalgOp.getLoc(), dyn_cast<MemRefType>(newSubviewType), input,
            offsets, shape, strides);
        linalgOp.setOperand(i, subview);
      }
    }
    return success();
  }

private:
  LinalgTilingOptions options;
};

void populateLinalgTilingPatterns(RewritePatternSet &patterns,
                                  LinalgTilingOptions options) {
  patterns.add<LinalgOpTiling>(patterns.getContext(), options);
}

struct LinalgTiling : public tpp::impl::LinalgTilingBase<LinalgTiling> {

  using LinalgTilingBase::LinalgTilingBase;

  void runOnOperation() override {
    LinalgTilingOptions options{mTileShape, nTileShape};
    RewritePatternSet patterns(&getContext());
    populateLinalgTilingPatterns(patterns, options);
    // GreedyRewriteConfig config;
    // config.strictMode = GreedyRewriteStrictness::ExistingOps;
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));

    // auto *parentOp = getOperation();
    /* SmallVector<ForallOp> innermostForAllloops;
    getInnermostForLoops(parentOp, innermostForAllloops);
    for (ForallOp loop : innermostForAllloops)
      if (succeeded(insertParallelLoop(loop, tileShapeM, tileShapeN))) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to tile the loop\n");
      }*/
  }
};
} // namespace tpp
} // namespace mlir
