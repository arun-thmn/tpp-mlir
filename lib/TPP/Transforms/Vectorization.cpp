//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements tile configuration hoisting on parallel loops.
//
//===----------------------------------------------------------------------===//
#include "TPP/Dialect/Xsmm/XsmmUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <iostream>
#include <list>

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_VECTORIZATIONPASS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;
using namespace std;

namespace mlir {
namespace tpp {

struct LinalgGenericToVector : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (xsmm::utils::getDataType(rewriter, linalgOp.getOperand(0).getType()) ==
            xsmm::DataTypeAttr::get(rewriter.getContext(),
                                    xsmm::DataType::BF16) &&
        linalgOp.getIteratorTypes().size() >= 5) {
      SmallVector<int64_t> shape;
      SmallVector<ReassociationIndices> indices;
      int index = 0;
      for (int i = 0; i < dyn_cast<ShapedType>(linalgOp.getOperand(0).getType())
                              .getShape()
                              .size();
           i++) {
        ReassociationIndices reassoc;
        if (i == dyn_cast<ShapedType>(linalgOp.getOperand(0).getType())
                         .getShape()
                         .size() -
                     1) {
          shape.push_back(dyn_cast<ShapedType>(linalgOp.getOperand(0).getType())
                              .getShape()[i] /
                          2);
          shape.push_back(2);
          reassoc.push_back(index++);
          reassoc.push_back(index++);
        } else {
          shape.push_back(dyn_cast<ShapedType>(linalgOp.getOperand(0).getType())
                              .getShape()[i]);
          reassoc.push_back(index++);
        }
        indices.push_back(reassoc);
      }
      auto expand = rewriter.create<memref::ExpandShapeOp>(
          linalgOp.getLoc(), shape, linalgOp.getOperand(0), indices);
      linalgOp.setOperand(0, expand.getResult());
      auto map0 = linalgOp.getIndexingMapsArray()[0];
      auto map1 = linalgOp.getIndexingMapsArray()[1];
      map0 = map0.insertResult(map1.getResult(map1.getNumResults() - 1), 3);
      map1 = map1.insertResult(
          dyn_cast<AffineBinaryOpExpr>(map1.getResult(1)).getLHS(), 2);
      map1 = map1.dropResult(1);
      linalgOp.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(
          {map0, map1, linalgOp.getIndexingMapsArray()[2]}));
    }
    return linalg::vectorize(rewriter, linalgOp);
  }
};

template <typename LinalgOp>
struct LinalgToVector : OpRewritePattern<LinalgOp> {
  using OpRewritePattern<LinalgOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    return linalg::vectorize(rewriter, linalgOp);
  }
};

struct VectorizationPass
    : public impl::VectorizationPassBase<VectorizationPass> {

  void populateCombinePatterns(RewritePatternSet &patterns) {
    patterns.add<LinalgToVector<linalg::BatchReduceMatmulOp>,
                 LinalgToVector<linalg::FillOp>>(patterns.getContext());
    patterns.add<LinalgGenericToVector>(patterns.getContext());
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateCombinePatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace tpp
} // namespace mlir
