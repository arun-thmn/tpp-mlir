//===-VectorContractToBF16DotProduct.cpp
//-----------------------------------------*-
// C++-*-===//
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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_BF16DOTPRODUCT
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace mlir {
namespace tpp {

static FailureOr<SmallVector<vector::TransferReadOp>>
getContractOperands(vector::ContractionOp contractOp) {
  SmallVector<vector::TransferReadOp> list;
  for (int i = 0; i < 3; i++) {
    auto vectorReadOp =
        contractOp.getOperand(i).getDefiningOp<vector::TransferReadOp>();
    if (!vectorReadOp)
      return failure();
    list.push_back(vectorReadOp);
  }
  return list;
}

static FailureOr<SmallVector<memref::SubViewOp>>
getReadOperands(SmallVector<vector::TransferReadOp> readOps) {
  SmallVector<memref::SubViewOp> list;
  for (vector::TransferReadOp readOp : readOps) {
    auto subViewOp = readOp.getOperand(0).getDefiningOp<memref::SubViewOp>();
    if (!subViewOp)
      return failure();
    list.push_back(subViewOp);
  }
  return list;
}

static FailureOr<SmallVector<scf::ForOp>>
getNestedLoop(vector::ContractionOp contractOp) {
  SmallVector<scf::ForOp> list;
  Operation *current = contractOp;
  for (int i = 0; i < 4; i++) {
    Operation *parent = current->getParentOfType<scf::ForOp>();
    if (!parent)
      return failure();
    list.push_back(dyn_cast<scf::ForOp>(parent));
    current = parent;
  }
  return list;
}

static LogicalResult checkNestedLoop(SmallVector<scf::ForOp> loops,
                                     SmallVector<memref::SubViewOp> subviews) {
  auto subviewOpLhsOffsets = subviews[0].getOffsets();
  auto subviewOpRhsOffsets = subviews[1].getOffsets();
  auto subviewOpAccOffsets = subviews[2].getOffsets();

  Value ivK = loops[0].getInductionVar();
  if (ivK != subviewOpLhsOffsets[2] || ivK != subviewOpRhsOffsets[1])
    return failure();

  Value ivReduction = loops[1].getInductionVar();
  if (ivReduction != subviewOpLhsOffsets[0] ||
      ivReduction != subviewOpRhsOffsets[0])
    return failure();

  Value ivN = loops[2].getInductionVar();
  if (ivN != subviewOpAccOffsets[1] || ivN != subviewOpRhsOffsets[2])
    return failure();

  Value ivM = loops[3].getInductionVar();
  if (ivM != subviewOpLhsOffsets[1] || ivM != subviewOpAccOffsets[0])
    return failure();

  return success();
}

struct BF16DotProductOp : OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {

    // Check the vector contract operation satisfies the required pattern.
    // Check the Acc, Lhs, and Rhs of contract operation

    auto operands = getContractOperands(contractOp);
    if (failed(operands))
      return rewriter.notifyMatchFailure(contractOp,
                                         "Invalid operands for contract op");
    auto readOps = *operands;
    auto vectorReadOpAcc = readOps[2];
    auto vectorReadOpLhs = readOps[0];
    auto vectorReadOpRhs = readOps[1];

    // Check whether the operand of vector transfer read is a subview
    auto readOpSubviews = getReadOperands(readOps);
    if (failed(readOpSubviews))
      return rewriter.notifyMatchFailure(
          contractOp, "Vector read op operands are not a subview");

    auto subviews = *readOpSubviews;
    auto subviewOpAcc = subviews[2];

    // Check the operation type MatMul, B-MatMul, or BR-MatMul (FP32/BF16)
    SmallVector<vector::IteratorType> contractIteratorTypes =
        contractOp.getIteratorTypesArray();
    int reductionCount =
        std::count(contractIteratorTypes.begin(), contractIteratorTypes.end(),
                   vector::IteratorType::reduction);

    if (reductionCount == 0)
      return rewriter.notifyMatchFailure(contractOp,
                                         "Matmul operation not supported yet");

    if (reductionCount == 1)
      return rewriter.notifyMatchFailure(
          contractOp, "Batch matmul operation not supported yet");

    if (reductionCount == 2)
      return rewriter.notifyMatchFailure(
          contractOp, "Batch reduce matmul operation without vnni layout");

    if (reductionCount > 3)
      return rewriter.notifyMatchFailure(
          contractOp, "The vector contract operation is not a gemm");

    auto lhsType = dyn_cast<ShapedType>(vectorReadOpLhs.getType());
    auto rhsType = dyn_cast<ShapedType>(vectorReadOpRhs.getType());
    auto accType = dyn_cast<ShapedType>(vectorReadOpAcc.getType());

    if (reductionCount == 3 &&
        (lhsType.getRank() != 4 || rhsType.getRank() != 4))
      return rewriter.notifyMatchFailure(
          contractOp,
          "Invalid rank for batch reduce operation with vnni layout");

    int64_t M = accType.getDimSize(0);
    int64_t N = accType.getDimSize(1);
    int64_t K = lhsType.getDimSize(lhsType.getRank() - 2);
    int64_t vnni = lhsType.getDimSize(lhsType.getRank() - 1);

    // K tile should be equal to vnni layout
    if (K != (vnni / 2))
      return rewriter.notifyMatchFailure(
          contractOp, "K tile size should be equal to VNNI layout");

    if (N != 32)
      return rewriter.notifyMatchFailure(
          contractOp,
          "N tile size should be equal to 32 to ensure avx512bf16 dp");

    if (vnni != 2)
      return rewriter.notifyMatchFailure(
          contractOp, "Only VNNI layout=2 is supported, now");

    auto loops = getNestedLoop(contractOp);
    if (failed(loops))
      return rewriter.notifyMatchFailure(
          contractOp, "Invalid loop nest in contract pattern");

    auto checkLoops = checkNestedLoop(*loops, subviews);
    if (failed(checkLoops))
      return rewriter.notifyMatchFailure(
          contractOp, "Loops doesn't match the iv in subviews");

    auto nestedLoops = *loops;
    auto kForOp = nestedLoops[0];
    auto reductionForOp = nestedLoops[1];

    rewriter.setInsertionPoint(reductionForOp);
    Value c0 =
        rewriter.create<arith::ConstantIndexOp>(reductionForOp.getLoc(), 0);
    auto elementType =
        (cast<MemRefType>(subviewOpAcc.getType())).getElementType();

    // Creating further subviews from the C matrix subview
    llvm::SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(K),
                                             rewriter.getIndexAttr(N)};
    llvm::SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1),
                                               rewriter.getIndexAttr(1)};
    llvm::SmallVector<Value, 8> subviewCMatrix;
    llvm::SmallVector<Value, 8> loopItrArgs;
    for (int i = 0; i < M; i++) {
      SmallVector<OpFoldResult> offsets = {
          rewriter.getIndexAttr(i),
          rewriter.getIndexAttr(0),
      };
      auto newSubview = rewriter.create<memref::SubViewOp>(
          reductionForOp.getLoc(), subviewOpAcc, offsets, sizes, strides);
      subviewCMatrix.push_back(newSubview);

      // vector <16xf32> for iterargs to accumulate results in fp32
      for (int j = 0; j < vnni; j++) {
        Value indexOp = rewriter.create<arith::ConstantIndexOp>(
            reductionForOp.getLoc(), j * (N / 2));
        auto valueCRow = rewriter.create<vector::LoadOp>(
            reductionForOp.getLoc(), VectorType::get({N / 2}, elementType),
            newSubview, ValueRange{c0, indexOp});
        auto bitcast_i16 = rewriter.create<vector::BitCastOp>(
            reductionForOp.getLoc(),
            VectorType::get({N / 2}, rewriter.getIntegerType(16)), valueCRow);
        auto extend_i32 = rewriter.create<arith::ExtUIOp>(
            reductionForOp.getLoc(),
            VectorType::get({N / 2}, rewriter.getIntegerType(32)), bitcast_i16);
        auto cst16 = rewriter.create<arith::ConstantOp>(
            reductionForOp.getLoc(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(32), N / 2));
        auto vectType = VectorType::get({N / 2}, rewriter.getIntegerType(32));
        auto shiftOp = rewriter.create<arith::ShLIOp>(
            reductionForOp.getLoc(), vectType, extend_i32,
            rewriter.create<vector::BroadcastOp>(reductionForOp.getLoc(),
                                                 vectType, cst16));
        auto f32CVector = rewriter.create<vector::BitCastOp>(
            reductionForOp.getLoc(),
            VectorType::get({N / 2}, rewriter.getF32Type()), shiftOp);

        loopItrArgs.push_back(f32CVector);
      }
    }

    SmallVector<Value, 8> bf16DP;

    // Code to re-create the reduction and k loop with iter args
    auto newReductionForOp = rewriter.create<scf::ForOp>(
        reductionForOp.getLoc(), reductionForOp.getLowerBound(),
        reductionForOp.getUpperBound(), reductionForOp.getStep(), loopItrArgs,
        [&](OpBuilder &rewriterNewReductionForOp, Location locNewReductionForOp,
            Value ivNewReductionForOp, ValueRange iterArgsNewReductionForOp) {
          auto newKForOp = rewriter.create<scf::ForOp>(
              kForOp.getLoc(), kForOp.getLowerBound(), kForOp.getUpperBound(),
              kForOp.getStep(), iterArgsNewReductionForOp,
              [&](OpBuilder &rewriterNewKForOp, Location locNewKForOp,
                  Value ivNewKForOp, ValueRange iterArgsNewKForOp) {
                IRMapping mapping;
                mapping.map(
                    vectorReadOpLhs.getSource().getDefiningOp()->getOperand(1),
                    ivNewReductionForOp);
                mapping.map(
                    vectorReadOpLhs.getSource().getDefiningOp()->getOperand(3),
                    ivNewKForOp);
                auto lhsClone = rewriterNewKForOp.clone(
                    *vectorReadOpLhs.getSource().getDefiningOp(), mapping);

                // Memory access for A Matrix into <32xbf16>
                llvm::SmallVector<Value, 8> vectorA;

                for (int i = 0; i < M; i++) {
                  Value indexOp = rewriter.create<arith::ConstantIndexOp>(
                      reductionForOp.getLoc(), i);
                  auto valueA = rewriterNewKForOp.create<vector::LoadOp>(
                      kForOp.getLoc(), VectorType::get({vnni}, elementType),
                      lhsClone->getResult(0), ValueRange{c0, indexOp, c0, c0});
                  auto bitcastValueA =
                      rewriterNewKForOp.create<vector::BitCastOp>(
                          kForOp.getLoc(),
                          VectorType::get({1}, rewriterNewKForOp.getI32Type()),
                          valueA);
                  auto broadcastValueA =
                      rewriterNewKForOp.create<vector::BroadcastOp>(
                          kForOp.getLoc(),
                          VectorType::get(16, rewriterNewKForOp.getI32Type()),
                          bitcastValueA);
                  auto bitcastValueA_32 =
                      rewriterNewKForOp.create<vector::BitCastOp>(
                          kForOp.getLoc(),
                          VectorType::get({N}, rewriterNewKForOp.getBF16Type()),
                          broadcastValueA);

                  vectorA.push_back(bitcastValueA_32);
                }

                IRMapping rhsMapping;
                rhsMapping.map(
                    vectorReadOpRhs.getSource().getDefiningOp()->getOperand(1),
                    ivNewReductionForOp);
                rhsMapping.map(
                    vectorReadOpRhs.getSource().getDefiningOp()->getOperand(2),
                    ivNewKForOp);
                auto rhsClone = rewriterNewKForOp.clone(
                    *vectorReadOpRhs.getSource().getDefiningOp(), rhsMapping);

                // Memory access for B Matrix into <32xbf16>
                llvm::SmallVector<Value, 8> vectorB;
                for (int i = 0, j = 0; i < vnni; i++, j = j + 16) {
                  Value indexOp = rewriter.create<arith::ConstantIndexOp>(
                      reductionForOp.getLoc(), j);
                  auto valueBRow = rewriterNewKForOp.create<vector::LoadOp>(
                      kForOp.getLoc(), VectorType::get({N}, elementType),
                      rhsClone->getResult(0), ValueRange{c0, c0, indexOp, c0});
                  vectorB.push_back(valueBRow);
                }

                // Code for x86vector.avx512.dot
                mlir::VectorType dstType =
                    mlir::VectorType::get({N / 2}, rewriter.getF32Type());
                for (int i = 0, k = 0; i < M; i++, k = k + vnni) {
                  for (int j = 0; j < vnni; j++) {
                    auto dp = rewriter.create<mlir::x86vector::DotBF16Op>(
                        kForOp.getLoc(), dstType, iterArgsNewKForOp[j + k],
                        vectorA[i], vectorB[j]);
                    bf16DP.push_back(dp);
                  }
                }

                rewriterNewKForOp.create<scf::YieldOp>(locNewKForOp, bf16DP);
              });

          rewriterNewReductionForOp.create<scf::YieldOp>(
              locNewReductionForOp, newKForOp.getResults());
        });

    // Downconvert <16xf32> to <16xbf16> and store into C Matrix
    for (int i = 0, k = 0; i < M; i++) {
      for (int j = 0; j < vnni; j++) {
        Value indexOp = rewriter.create<arith::ConstantIndexOp>(
            reductionForOp.getLoc(), j * 16);
        auto bf16vec = rewriter.create<arith::TruncFOp>(
            reductionForOp.getLoc(),
            VectorType::get({16}, rewriter.getBF16Type()),
            newReductionForOp.getResult(k));
        rewriter.create<vector::StoreOp>(reductionForOp.getLoc(), bf16vec,
                                         subviewCMatrix[i],
                                         ValueRange{c0, indexOp});
        k++;
      }
    }

    // Delete the contract operation
    for (auto result : contractOp->getResults()) {
      for (auto *userOp : result.getUsers()) {
        rewriter.eraseOp(userOp);
      }
    }
    rewriter.eraseOp(contractOp);
    return success();
  }
};

void populateBF16DotProductPatterns(RewritePatternSet &patterns) {
  patterns.add<BF16DotProductOp>(patterns.getContext());
}

struct BF16DotProduct : public impl::BF16DotProductBase<BF16DotProduct> {
  using BF16DotProductBase::BF16DotProductBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateBF16DotProductPatterns(patterns);
    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;
    (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
  }
};
} // namespace tpp
} // namespace mlir
