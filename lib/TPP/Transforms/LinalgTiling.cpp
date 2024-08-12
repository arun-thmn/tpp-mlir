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

#define DEBUG_TYPE "linalg-tiling"

namespace mlir {
namespace tpp {
#define GEN_PASS_DECL_LINALGTILINGPASS
#define GEN_PASS_DEF_LINALGTILINGPASS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;
using namespace mlir::scf;
using namespace std;

namespace mlir {
namespace tpp {


template <typename LinalgOp>
struct LinalgTiling : OpRewritePattern<LinalgOp> {
  using OpRewritePattern<LinalgOp>::OpRewritePattern;

  LinalgTiling(MLIRContext *ctx, LinalgTilingOptions options)
      : OpRewritePattern<LinalgOp>(ctx), options(options) {}

  LogicalResult matchAndRewrite(LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {

  SmallVector<unsigned> tileShapeM = options.mTileShape;
  SmallVector<unsigned> tileShapeN = options.nTileShape;

  SmallVector<SmallVector<unsigned>> tiles = {tileShapeM, tileShapeN};
  SmallVector<SmallVector<unsigned>> tilingVectors{{tileShapeM}, {tileShapeN}};

  rewriter.setInsertionPointToStart(op.getBody());
    SmallVector<SmallVector<unsigned>> tileQuotient;
     auto contractionDim = inferContractionDims(linalgop);
                 auto reductionDim = contractionDim->k.front();
    for (size_t i = 0; i < linalgop.getNumOperands(); i++) {
	SmallVector<unsigned> indecies;
      auto input = linalgop.getOperand(i);
          auto operandType = input.getType();
          SmallVector<int64_t> shape;
          size_t index = 0;
          for (size_t j = 0;
               j < dyn_cast<ShapedType>(operandType).getShape().size(); j++) {
		  if (dyn_cast<ShapedType>(operandType).getShape()[j] == reductionDim) {
                                indecies.push_back(1);
		  }

  		for (size_t k = 0; k < tiles[i]; k++) {
			indecies.push_back(dyn_cast<ShapedType>(operandType).getShape()[j]/tiles[i][k])
			break;
		}
		linalgop.setOperand(i, MemRefType::get({indices},rewriter.getIndexType()));
		tileQuotient.push_back(indecies);
	  }
    }
	llvm::SmallDenseSet<int64_t> intersection;
	for (size_t i = 0; i < linalgop.getNumOperands(); i++) {
		auto parPerm = findPermutationsIndexingOperand(linalgop.getIndexingMapsArray()[i], linalgop.getIterators(), par);
		auto redPerm = findPermutationsIndexingOperand(linalgop.getIndexingMapsArray()[i], linalgop.getIterators(), red);
		auto unionPerm = lvm::set_union(parPerm, redPerm);
		if (i == 0) {
			intersection = unionPerm;
		}
			else {

		intersection = llvm::set_intersect(intersection, unionPerm);
			}
	}


  }
  return success();
}

static bool getInnermostForLoops(Operation *rootOp,
                                 SmallVectorImpl<scf::ForallOp> &result) {
  assert(rootOp != nullptr && "Root operation must not be a nullptr.");
  bool rootEnclosesForAllloops = false;
  for (Region &region : rootOp->getRegions()) {
    for (Block &block : region.getBlocks()) {
      for (Operation &op : block) {
        bool enclosesPloops = getInnermostForLoops(&op, result);
        rootEnclosesForAllloops |= enclosesPloops;
        if (auto ploop = dyn_cast<scf::ForallOp>(op)) {
          rootEnclosesForAllloops = true;

          // Collect forall loop if it is an innermost one.
          if (!enclosesPloops)
            result.push_back(ploop);
        }
      }
    }
  }
  return rootEnclosesForAllloops;
}

struct VectorTilingPass
    : public tpp::impl::VectorTilingPassBase<VectorTilingPass> {

  using VectorTilingPassBase::VectorTilingPassBase;

  void runOnOperation() override {
    auto *parentOp = getOperation();
    SmallVector<ForallOp> innermostForAllloops;
    getInnermostForLoops(parentOp, innermostForAllloops);
    for (ForallOp loop : innermostForAllloops)
      if (succeeded(insertParallelLoop(loop, tileShapeM, tileShapeN))) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to tile the loop\n");
      }
  }
};
} // namespace tpp
} // namespace mlir
