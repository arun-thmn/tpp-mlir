//===- IntelAMXTileConfig.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file inserts tile configuration calls.
//
//===----------------------------------------------------------------------===//
#include "TPP/Dialect/Xsmm/XsmmOps.h"
#include "TPP/Dialect/Xsmm/XsmmUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_INTELAMXTILECONFIGINSERTIONPASS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;

namespace mlir {
namespace tpp {

template <typename VectorOp>
static DenseI64ArrayAttr getDispatchInputs(PatternRewriter &rewriter,
                                           VectorOp op, Operation *&firstOp,
                                           Operation *&lastOp) {
  FailureOr<xsmm::BrgemmInfo> info;
  SmallVector<std::function<bool(Operation * op)>> operations;
  operations.push_back(xsmm::FuncType<VectorOp>);
  SmallVector<Operation *> opChain;
  xsmm::utils::WithOps(&op->getParentOp()->getRegion(0), op->getParentOp(),
                       operations, opChain);
  assert(opChain[0] == op);

  SmallVector<std::function<bool(Operation * op)>> inputOperations;
  inputOperations.push_back(xsmm::FuncType<vector::TransferReadOp>);
  inputOperations.push_back(xsmm::FuncType<vector::TransferReadOp>);
  inputOperations.push_back(xsmm::FuncType<vector::TransferReadOp>);
  SmallVector<OpOperand *> inputs;
  SmallVector<Operation *> inputOpChain;
  xsmm::utils::WithInputs(rewriter, op, inputOperations, inputs, inputOpChain);
  operations.clear();
  operations.push_back(xsmm::FuncType<arith::AddFOp>);
  opChain.clear();
  if (xsmm::utils::WithOps(&op->getParentOp()->getRegion(0), op->getParentOp(),
                           operations, opChain)) {
    auto biasAdd = opChain[0];
    assert(isa<arith::AddFOp>(biasAdd));
    SmallVector<std::function<bool(Operation * op)>> inputOperations;
    inputOperations.push_back(xsmm::FuncType<vector::BroadcastOp>);
    inputOperations.push_back(xsmm::FuncType<vector::TransferReadOp>);
    SmallVector<OpOperand *> addInputs;
    SmallVector<Operation *> addOpChain;
    if (xsmm::utils::WithInputs(rewriter, biasAdd, inputOperations, addInputs,
                                addOpChain)) {
      SmallVector<OpOperand *> addOutputs;
      SmallVector<Operation *> addOutputChain;
      if (xsmm::utils::WithOutput(biasAdd,
                                  xsmm::FuncType<vector::TransferWriteOp>,
                                  addOutputs, addOutputChain)) {
        operations.clear();
        operations.push_back(xsmm::FuncType<arith::MaximumFOp>);
        opChain.clear();
        if (xsmm::utils::WithOps(&op->getParentOp()->getRegion(0),
                                 op->getParentOp(), operations, opChain)) {
          auto maxF = opChain[0];
          assert(isa<arith::MaximumFOp>(maxF));
          SmallVector<OpOperand *> maxfInputs;
          SmallVector<std::function<bool(Operation * op)>> maxFOperations;
          maxFOperations.push_back(xsmm::FuncType<vector::TransferReadOp>);
          SmallVector<Operation *> maxfOpChain;
          if (xsmm::utils::WithInputs(rewriter, maxF, maxFOperations,
                                      maxfInputs, maxfOpChain)) {
            SmallVector<OpOperand *> maxfOutputs;
            SmallVector<Operation *> maxfOutputChain;
            if (xsmm::utils::WithOutput(maxF,
                                        xsmm::FuncType<vector::TransferWriteOp>,
                                        maxfOutputs, maxfOutputChain)) {
              SmallVector<Operation *> contractOutputChain;
              SmallVector<OpOperand *> outputs;
              if (xsmm::utils::WithOutput(
                      op, xsmm::FuncType<vector::TransferWriteOp>, outputs,
                      contractOutputChain)) {
                info = xsmm::utils::isMappableToBrgemm(
                    rewriter, op, inputs, outputs, op.getIndexingMapsArray());
                vector::TransferWriteOp zeroOp;
                xsmm::utils::WithZeroInit(inputs[2], zeroOp);
                if (zeroOp) {
                  firstOp = zeroOp;
                } else {
                  firstOp = inputOpChain[0];
                }

                lastOp = maxfOutputChain[0];
              }
            }
          }
        }
      }
    }
  } else {
    SmallVector<OpOperand *> outputs;
    SmallVector<Operation *> outputChain;
    if (xsmm::utils::WithOutput(op, xsmm::FuncType<vector::TransferWriteOp>,
                                outputs, outputChain)) {

      vector::TransferWriteOp zeroOp;
      xsmm::utils::WithZeroInit(inputs[2], zeroOp);
      info = xsmm::utils::isMappableToBrgemm(rewriter, op, inputs, outputs,
                                             op.getIndexingMapsArray());
      if (zeroOp) {
        firstOp = zeroOp;
      } else {
        firstOp = inputOpChain[0];
      }
      lastOp = outputChain[0];
    }
  }

  auto brgemmInfo = *info;
  auto m = brgemmInfo.m;
  auto n = brgemmInfo.n;
  auto k = brgemmInfo.k;
  int64_t lda = brgemmInfo.lda;
  int64_t ldb = brgemmInfo.ldb;
  int64_t ldc = brgemmInfo.ldc;
  int64_t strideA = brgemmInfo.strideA;
  int64_t strideB = brgemmInfo.strideB;
  DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
      rewriter.getContext(),
      ArrayRef<int64_t>{m, n, k, lda, ldb, ldc, strideA, strideB});
  return dims;
}

template <typename VectorOp>
struct IntelAMXTileConfig : OpRewritePattern<VectorOp> {
  using OpRewritePattern<VectorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(VectorOp op,
                                PatternRewriter &rewriter) const override {
    if (xsmm::utils::getDataType(rewriter, op.getOperand(1).getType()) !=
        xsmm::DataTypeAttr::get(rewriter.getContext(), xsmm::DataType::BF16))
      return failure();
    Operation *definingOp = &*op;
    if (definingOp->hasAttr(xsmm::stringifyGemmFlags(
            mlir::xsmm::GemmFlags::NO_RESET_TILECONFIG)) ||
        definingOp->hasAttr(xsmm::stringifyGemmFlags(
            mlir::xsmm::GemmFlags::NO_SETUP_TILECONFIG))) {
      return failure();
    }
    SmallVector<Attribute> attributesSetup;
    attributesSetup.push_back(xsmm::GemmFlagsAttr::get(
        rewriter.getContext(), xsmm::GemmFlags::NO_RESET_TILECONFIG));
    ArrayAttr invokeFlags = rewriter.getArrayAttr(attributesSetup);
    Operation *firstOp = nullptr, *lastOp = nullptr;
    auto dispatchInputs = getDispatchInputs(rewriter, op, firstOp, lastOp);
    assert(firstOp != nullptr && lastOp != nullptr);
    rewriter.setInsertionPoint(firstOp);
    auto tileConfigSetup = rewriter.create<xsmm::IntelAMXTileConfigDispatchOp>(
        op.getLoc(), rewriter.getI64Type(),
        DenseI64ArrayAttr::get(rewriter.getContext(), dispatchInputs),
        invokeFlags,
        xsmm::utils::getDataType(rewriter, op.getOperand(1).getType()));

    SmallVector<Attribute> attributesReset;
    attributesReset.push_back(xsmm::GemmFlagsAttr::get(
        rewriter.getContext(), xsmm::GemmFlags::NO_SETUP_TILECONFIG));

    auto tileConfigReset = rewriter.create<xsmm::IntelAMXTileConfigDispatchOp>(
        op.getLoc(), rewriter.getI64Type(),
        DenseI64ArrayAttr::get(rewriter.getContext(), dispatchInputs),
        rewriter.getArrayAttr(attributesReset),
        xsmm::utils::getDataType(rewriter, op.getOperand(1).getType()));

    definingOp->setAttr(
        xsmm::stringifyGemmFlags(xsmm::GemmFlags::NO_RESET_TILECONFIG),
        xsmm::GemmFlagsAttr::get(rewriter.getContext(),
                                 xsmm::GemmFlags::NO_RESET_TILECONFIG));
    definingOp->setAttr(
        xsmm::stringifyGemmFlags(xsmm::GemmFlags::NO_SETUP_TILECONFIG),
        xsmm::GemmFlagsAttr::get(rewriter.getContext(),
                                 xsmm::GemmFlags::NO_SETUP_TILECONFIG));

    auto alloca = rewriter.create<memref::AllocaOp>(
        op.getLoc(), MemRefType::get({64}, rewriter.getI8Type()));

    ValueRange tileConfigInputs{alloca};
    rewriter.create<mlir::xsmm::IntelAMXTileConfigOp>(
        op.getLoc(), tileConfigSetup, tileConfigInputs);

    rewriter.setInsertionPointAfter(lastOp);
    ValueRange tileResetInputs{alloca};
    rewriter.create<mlir::xsmm::IntelAMXTileConfigOp>(
        op.getLoc(), tileConfigReset, tileResetInputs);

    return success();
  }
};

struct IntelAMXTileConfigInsertionPass
    : public impl::IntelAMXTileConfigInsertionPassBase<
          IntelAMXTileConfigInsertionPass> {
  void populateCombinePatterns(RewritePatternSet &patterns) {
    patterns.add<IntelAMXTileConfig<vector::ContractionOp>>(
        patterns.getContext());
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateCombinePatterns(patterns);
    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                       config);
  }
};
} // namespace tpp
} // namespace mlir
