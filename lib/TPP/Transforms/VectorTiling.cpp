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

#define DEBUG_TYPE "vector-tiling"

namespace mlir {
namespace tpp {
#define GEN_PASS_DECL_VECTORTILINGPASS
#define GEN_PASS_DEF_VECTORTILINGPASS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;
using namespace mlir::scf;
using namespace std;

namespace mlir {
namespace tpp {

static memref::SubViewOp
insertSubview(ArrayRef<int64_t> tensorShape, Type type, MemRefType resultType,
              SmallVector<ReassociationIndices> reassociation, ForallOp op,
              Value operand, IRRewriter &rewriter,
              SmallVector<OpFoldResult> offsets,
              SmallVector<OpFoldResult> sizes, Operation *xsmmOp) {
  OpBuilder::InsertionGuard guard(rewriter);

  rewriter.setInsertionPoint(op);
  auto expandShape = rewriter.create<memref::ExpandShapeOp>(
      op.getLoc(),
      MemRefType::get({tensorShape},
                      dyn_cast<MemRefType>(type).getElementType()),
      operand, reassociation);
  expandShape.setStaticOutputShape(tensorShape);

  rewriter.setInsertionPoint(xsmmOp);
  SmallVector<int64_t> tileSizes;

  tileSizes.append(dyn_cast<ShapedType>(resultType).getShape().begin(),
                   dyn_cast<ShapedType>(resultType).getShape().end());

  auto subviewType =
      MemRefType::get({tileSizes}, dyn_cast<MemRefType>(type).getElementType());

  auto [originalStride, originalOffset] =
      getStridesAndOffset(dyn_cast<MemRefType>(resultType));
  subviewType = MemRefType::get(
      {tileSizes}, dyn_cast<MemRefType>(subviewType).getElementType(),
      StridedLayoutAttr::get(rewriter.getContext(), ShapedType::kDynamic,
                             originalStride));

  SmallVector<OpFoldResult> strides(sizes.size(), rewriter.getIndexAttr(1));

  auto subview = rewriter.create<memref::SubViewOp>(
      op.getLoc(), dyn_cast<MemRefType>(subviewType), expandShape.getResult(),
      offsets, sizes, strides);
  return subview;
}

static LogicalResult insertParallelLoop(ForallOp op, unsigned mTileShape,
                                        unsigned nTileShape) {
  OpBuilder b(op);
  IRRewriter rewriter(b.getContext());
  if (mTileShape == 0 || nTileShape == 0) {
    LLVM_DEBUG(llvm::dbgs() << "require tile shape to not be zero");
    return failure();
  }
  SmallVector<Operation *> vectorOpList;
  for (auto operItr = op.getBody()->begin(); operItr != op.getBody()->end();
       operItr++) {
    Operation *oper = &*operItr;
    if (dyn_cast<vector::ContractionOp>(oper)) {
      vectorOpList.push_back(&*oper);
    }
  }

  if (vectorOpList.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "require vector op in loop");
    return failure();
  }
  int mSize = (op.getStaticUpperBound()[0] - op.getStaticLowerBound()[0]) /
              op.getStaticStep()[0];
  int nSize = (op.getStaticUpperBound()[1] - op.getStaticLowerBound()[1]) /
              op.getStaticStep()[1];
  SmallVector<unsigned> tileShapeM;
  SmallVector<unsigned> tileShapeN;

  tileShapeM.push_back(mTileShape);
  if (mSize % mTileShape != 0) {
    LLVM_DEBUG(llvm::dbgs() << "require m tile shape to match tensor shape");
    return failure();
  }
  tileShapeM.push_back(mSize / mTileShape);

  tileShapeN.push_back(nTileShape);
  if (nSize % nTileShape != 0) {
    LLVM_DEBUG(llvm::dbgs() << "require n tile shape to match tensor shape");
    return failure();
  }
  tileShapeN.push_back(nSize / nTileShape);

  SmallVector<Value> oldArgs(op.getBody()->getArguments().begin(),
                             op.getBody()->getArguments().end());

  int boundSize = tileShapeM.size() + tileShapeN.size();

  // Set the new bounds of for loop
  SmallVector<int64_t> lbs(boundSize, 0), steps(boundSize, 1);

  SmallVector<int64_t> ubs(tileShapeM.begin(), tileShapeM.end());
  ubs.append(tileShapeN.begin(), tileShapeN.end());

  op.setStaticLowerBound(lbs);
  op.setStaticUpperBound(ubs);
  op.setStaticStep(steps);

  // Add new induction var args to the for loop
  int numArgs = op.getBody()->getArguments().size();

  for (int i = 0; i < boundSize - numArgs; i++)
    op.getBody()->addArgument(b.getIndexType(), op.getLoc());

  SmallVector<int64_t> tileOffsets{
      0, static_cast<int64_t>(tileShapeM.size()),
      static_cast<int64_t>(tileShapeM.size() + tileShapeN.size())};

  SmallVector<SmallVector<unsigned>> tilingVectors{{tileShapeM}, {tileShapeN}};

  rewriter.setInsertionPointToStart(op.getBody());

  SmallVector<Operation *> expandedInputs;
  for (size_t k = 0; k < vectorOpList.size(); k++) {
    auto operation = vectorOpList[k];
    for (size_t i = 0; i < operation->getNumOperands(); i++) {
      auto input = operation->getOperand(i);
      if (input.getDefiningOp() != NULL) {
        auto definingOp = input.getDefiningOp()->getOperand(0);
        Operation *subviewOp = definingOp.getDefiningOp();
        if (isa<memref::ExpandShapeOp>(subviewOp))
          subviewOp = subviewOp->getOperand(0).getDefiningOp();
        if (find(expandedInputs.begin(), expandedInputs.end(), subviewOp) ==
                expandedInputs.end() &&
            !isa<memref::GetGlobalOp>(subviewOp)) {
          Value operand = subviewOp->getOperand(0);
          auto operandType = operand.getType();
          auto resultType =
              dyn_cast<MemRefType>(subviewOp->getResult(0).getType());
          SmallVector<int64_t> shape;
          SmallVector<ReassociationIndices> indices;
          size_t index = 0;
          unsigned opCount = 1;
          SmallVector<OpFoldResult> matchedInductionVariables, sizes;
          for (size_t j = 0;
               j < dyn_cast<ShapedType>(operandType).getShape().size(); j++) {
            if (j < subviewOp->getNumOperands() - 1) {
              size_t matchedIndex = j;
              bool matchedInductionVar = false;
              if (opCount < subviewOp->getNumOperands()) {
                for (size_t oldOpCount = j; oldOpCount < oldArgs.size();
                     oldOpCount++) {
                  if (oldArgs[oldOpCount] == subviewOp->getOperand(opCount)) {
                    matchedInductionVar = true;
                    matchedIndex = oldOpCount;
                    break;
                  }
                }
                unsigned tileSize = 1;
                auto cumulativeTileSize =
                    getCumulativeTileSize(tilingVectors, matchedIndex);
                ReassociationIndices reassociationIndex;
                shape.append(tilingVectors[matchedIndex].begin(),
                             tilingVectors[matchedIndex].end());
                for (size_t m = 0; m < tilingVectors[matchedIndex].size();
                     m++) {
                  reassociationIndex.push_back(index++);
                  sizes.push_back(rewriter.getIndexAttr(1));
                  tileSize *= tilingVectors[matchedIndex][m];
                  if (matchedInductionVar)
                    matchedInductionVariables.push_back(op.getInductionVar(
                        matchedIndex * cumulativeTileSize + m));
                  else {
                    if (opCount < subviewOp->getNumOperands()) {
                      if (m == 0) {
                        auto val = subviewOp->getOperand(opCount);
                        if (val != NULL && val.getDefiningOp() != NULL &&
                            isa<affine::AffineApplyOp>(val.getDefiningOp())) {
                          for (int n = 0;
                               n < val.getDefiningOp()->getNumOperands(); n++) {
                            for (size_t k = 0; k < oldArgs.size(); k++) {
                              if (oldArgs[k] ==
                                  val.getDefiningOp()->getOperand(n)) {
                                auto cumulativeTileSize =
                                    getCumulativeTileSize(tilingVectors, k);
                                for (size_t l = 0; l < tilingVectors[k].size();
                                     l++) {
                                  matchedInductionVariables.push_back(
                                      op.getInductionVar(
                                          k * cumulativeTileSize + l));
                                }
                              }
                            }
                          }
                        }
                      }
                    } else {
                      matchedInductionVariables.push_back(
                          rewriter.getIndexAttr(0));
                    }
                  }
                }

                if (tileSize <
                    dyn_cast<ShapedType>(operandType).getShape()[j]) {
                  auto spill = dyn_cast<ShapedType>(operandType).getShape()[j] /
                               tileSize;
                  shape.push_back(spill);
                  sizes.push_back(rewriter.getIndexAttr(spill));
                  reassociationIndex.push_back(index++);
                  matchedInductionVariables.push_back(rewriter.getIndexAttr(0));
                }
                indices.push_back(reassociationIndex);
                opCount++;
              }
            } else {
              ReassociationIndices reassociationIndex;
              reassociationIndex.push_back(index++);
              indices.push_back(reassociationIndex);
              auto resultTensor =
                  dyn_cast<ShapedType>(operandType).getShape()[j];
              shape.push_back(resultTensor);
              sizes.push_back(rewriter.getIndexAttr(resultTensor));
              matchedInductionVariables.push_back(rewriter.getIndexAttr(0));
            }
          }

          memref::SubViewOp createdSubviewOp = insertSubview(
              shape, operandType, resultType, indices, op, operand, rewriter,
              matchedInductionVariables, sizes, subviewOp);

          subviewOp->getResult(0).replaceAllUsesWith(
              createdSubviewOp.getResult());
          expandedInputs.push_back(createdSubviewOp);
        }
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
