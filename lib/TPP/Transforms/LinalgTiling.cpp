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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/Debug.h"
#include <iostream>
#include <list>

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"



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

//	namespace {

template <typename LinalgOp>
struct LinalgOpTiling : public OpRewritePattern<LinalgOp> {
  using OpRewritePattern<LinalgOp>::OpRewritePattern;

  LinalgOpTiling(MLIRContext *ctx, LinalgTilingOptions tilingoptions)
      : OpRewritePattern<LinalgOp>(ctx), options(tilingoptions) {}

  LogicalResult matchAndRewrite(LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {

  auto tileShapeM = options.mTileShape;
  auto tileShapeN = options.nTileShape;

  SmallVector<ArrayRef<unsigned int>> tiles = {tileShapeM, tileShapeN};

  rewriter.setInsertionPointToStart(linalgOp.getBody());
    SmallVector<SmallVector<int64_t>> tileQuotient;
     auto contractionDim = inferContractionDims(linalgOp);
                 auto reductionDim = contractionDim->k.front();
    for (size_t i = 0; i < linalgOp.getNumOperands(); i++) {
	SmallVector<int64_t> indecies;
	
      auto input = linalgOp.getOperand(i);
          auto operandType = input.getType();
          SmallVector<int64_t> shape;
          size_t index = 0;
          for (size_t j = 0;
               j < dyn_cast<ShapedType>(operandType).getShape().size(); j++) {
		  if (dyn_cast<ShapedType>(operandType).getShape()[j] == reductionDim) {
                                indecies.push_back(1);
		  }

  		for (size_t k = 0; k < tiles[i].size(); k++) {
			indecies.push_back(dyn_cast<ShapedType>(operandType).getShape()[j]/tiles[i][k]);
			break;
		}
		linalgOp.setOperand(i, MemRefType::get(ArrayRef(indecies),rewriter.getIndexType()));
		tileQuotient.push_back(indecies);
	  }
    }
	llvm::SmallDenseSet<int64_t> intersection;
	for (size_t i = 0; i < linalgOp.getNumOperands(); i++) {
		auto parPerm = findPermutationsIndexingOperand(linalgOp.getIndexingMapsArray()[i], linalgOp.getIterators(), par);
		auto redPerm = findPermutationsIndexingOperand(linalgOp.getIndexingMapsArray()[i], linalgOp.getIterators(), red);
		auto unionPerm = set_union(parPerm, redPerm);
		if (i == 0) {
			intersection = unionPerm;
		}
			else {
		intersection = set_intersection(intersection, unionPerm);
			}
	}
	return success();
  }

	private:
  	LinalgTilingOptions options;
};


void populateLinalgTilingPatterns(RewritePatternSet &patterns,
                                 LinalgTilingOptions options) {
  patterns.add<LinalgOpTiling>(
      patterns.getContext(), options);
}


struct LinalgTiling
    : public tpp::impl::LinalgTilingBase<LinalgTiling> {

  using LinalgTilingBase::LinalgTilingBase;

  void runOnOperation() override {
    LinalgTilingOptions options{mTileShape, nTileShape};
    RewritePatternSet patterns(&getContext());
     populateLinalgTilingPatterns(patterns, options);
    //GreedyRewriteConfig config;
    //config.strictMode = GreedyRewriteStrictness::ExistingOps;
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));

    //auto *parentOp = getOperation();
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
