//===- ConvertTppToLoops.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// NOTE copy from IREE so that I can use linalgx.pack and unpack.
// See: ConvertToLoops.cpp

#include "TPP/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

/// Recursive method that lowers one dimension of the `TiledOpInterface` to
/// scalar loops at a time.
static LogicalResult lowerToLoopsImpl(OpBuilder &builder,
                                      TilingInterface tilableOp,
                                      ArrayRef<Range> loopRanges,
                                      unsigned loopDepth,
                                      SmallVectorImpl<Value> &ivs) {
  Location loc = tilableOp.getLoc();
  if (loopDepth == loopRanges.size()) {
    return tilableOp.generateScalarImplementation(builder, loc, ivs);
  }
  LogicalResult status = success();
  builder.create<scf::ForOp>(
      loc,
      getValueOrCreateConstantIndexOp(builder, loc,
                                      loopRanges[loopDepth].offset),
      getValueOrCreateConstantIndexOp(builder, loc, loopRanges[loopDepth].size),
      getValueOrCreateConstantIndexOp(builder, loc,
                                      loopRanges[loopDepth].stride),
      ValueRange{}, [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
        ivs.push_back(iv);
        status = lowerToLoopsImpl(b, tilableOp, loopRanges, loopDepth + 1, ivs);
        b.create<scf::YieldOp>(loc);
      });
  return status;
}

/// Main entry point for lowering `TiledOpInterface` op to loops.
static LogicalResult lowerToLoops(OpBuilder &builder,
                                  TilingInterface tilableOp) {
  SmallVector<Range> loopBounds = tilableOp.getIterationDomain(builder);
  SmallVector<Value> ivs;
  return lowerToLoopsImpl(builder, tilableOp, loopBounds, 0, ivs);
}

/// Pattern rewriter hook to lower a `TiledOpInterface` to loops.
namespace {
struct TilingInterfaceLowerToLoopsPattern : public RewritePattern {
  TilingInterfaceLowerToLoopsPattern(MLIRContext *context,
                                     PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto tilableOp = dyn_cast<TilingInterface>(op);
    if (!tilableOp) {
      return rewriter.notifyMatchFailure(op, "not TilingInterface op");
    }
    // Avoid handling `LinalgOp`s here for now. Eventually this should
    // be able to handle everything (or this pass would be deprecated to use
    // something upstream).
    if (isa<linalg::LinalgOp>(op)) {
      return rewriter.notifyMatchFailure(op, "ignoring LinalgOps");
    }
    // if (llvm::any_of(tilableOp->getResults(),
    //                  [&](Value v) { return v.getType().isa<ShapedType>(); }))
    //                  {
    //   return rewriter.notifyMatchFailure(
    //       tilableOp, "lower to loops needs to have tensor semantics");
    // }
    if (failed(lowerToLoops(rewriter, tilableOp))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {
struct LinalgExtToLoopsPass
    : public LinalgExtToLoopsBase<LinalgExtToLoopsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, func::FuncDialect,
                    mlir::arith::ArithDialect, memref::MemRefDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.insert<TilingInterfaceLowerToLoopsPattern>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createLinalgXToLoopsPass() {
  return std::make_unique<LinalgExtToLoopsPass>();
}