//===-HoistVectorTransfers.cpp -----------------------------------------*-
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
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

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

    auto subViews = getReadOperands(readOps);
    auto subviews = *subViews;
    auto subviewOpLhs = subviews[0];
    auto subviewOpRhs = subviews[1];
    auto subviewOpAcc = subviews[2];
    auto elementType = (cast<MemRefType>(subviewOpAcc.getType())).getElementType();

    auto lhsType = dyn_cast<ShapedType>(vectorReadOpLhs.getType());
    auto rhsType = dyn_cast<ShapedType>(vectorReadOpRhs.getType());
    auto accType = dyn_cast<ShapedType>(vectorReadOpAcc.getType());

    int64_t M = accType.getDimSize(0);
    int64_t N = accType.getDimSize(1); 
    int64_t K = lhsType.getDimSize(lhsType.getRank() - 2);
    int64_t vnni = lhsType.getDimSize(lhsType.getRank() - 1);

    if (lhsType.getRank() == 3)
	    return failure();

    if (K != 1)
	    return failure();

    if (N != 32)
	    return failure();

    
    auto loops = getNestedLoop(contractOp); 
    auto nestedLoops = *loops;
    auto kForOp = nestedLoops[0];
    auto reductionForOp = nestedLoops[1];

    // Move the vector transfer read before the reduction and k loop
    rewriter.setInsertionPoint(reductionForOp);
    
    SmallVector<OpFoldResult> mixedSizes = {rewriter.getIndexAttr(K),
                                            rewriter.getIndexAttr(N)};
    SmallVector<OpFoldResult> mixedStrides = {rewriter.getIndexAttr(1),
                                              rewriter.getIndexAttr(1)};

    SmallVector<Value, 4> subview_2_splits;
    for (int i = 0; i < M; i++) {
      SmallVector<OpFoldResult> mixedOffsets = {
          rewriter.getIndexAttr(i),
          rewriter.getIndexAttr(0),
      };
      auto split = rewriter.create<memref::SubViewOp>(
          reductionForOp.getLoc(), subviewOpAcc, mixedOffsets, mixedSizes, mixedStrides);
      subview_2_splits.push_back(split);
    }

    auto elem = rewriter.create<mlir::arith::ConstantOp>(reductionForOp.getLoc(), rewriter.getF32Type(), rewriter.getFloatAttr(rewriter.getF32Type(), 0.0));
    auto bcast = rewriter.create<vector::BroadcastOp>(
                      reductionForOp.getLoc(), VectorType::get(16, elem.getType()), elem);

    SmallVector<Value, 4> initAccs;
    for (int i = 0; i < (M*vnni); i++) {
      initAccs.push_back(bcast);
    }
    
    Value c0 = rewriter.create<arith::ConstantIndexOp>(reductionForOp.getLoc(), 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(reductionForOp.getLoc(), 1);

    SmallVector<Value, 4> bf16DP;
    // Code to re-create the reduction and k loop with iter args
    auto newReductionForOp = rewriter.create<scf::ForOp>(
        reductionForOp.getLoc(), reductionForOp.getLowerBound(),
        reductionForOp.getUpperBound(), reductionForOp.getStep(),
        initAccs,
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



		SmallVector<Value, 4> broadcasts;
                for (int i = 0; i < M; i++) {
                  auto elem1 = rewriterNewKForOp.create<memref::LoadOp>(
                      kForOp.getLoc(), lhsClone->getResult(0),
                      ValueRange{
                          c0,
                          rewriterNewKForOp.create<arith::ConstantIndexOp>(kForOp.getLoc(), i),
                          c0, c0});
		  auto elem2 = rewriterNewKForOp.create<memref::LoadOp>(
                      kForOp.getLoc(), lhsClone->getResult(0),
                      ValueRange{
                          c0,
                          rewriterNewKForOp.create<arith::ConstantIndexOp>(kForOp.getLoc(), i),
                          c0, c1});
                  auto bcast1 = rewriterNewKForOp.create<vector::BroadcastOp>(
                      kForOp.getLoc(), VectorType::get(16, elem1.getType()), elem1);
		  auto bcast2 = rewriterNewKForOp.create<vector::BroadcastOp>(
                      kForOp.getLoc(), VectorType::get(16, elem2.getType()), elem2);

		//llvm::SmallVector<int64_t, 16> mask = {0, 32, 1, 32, 2, 32, 3, 32, 4, 32, 5, 32, 6, 32, 7, 32, 8, 32, 9, 32, 10, 32, 11, 32, 12, 32, 13, 32, 14, 32, 15, 32};
		llvm::SmallVector<int64_t, 16> mask = {0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39, 8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47};
	          llvm::SmallVector<int64_t, 16> mask2 = {32, 0, 32, 1, 32, 2, 32, 3, 32, 4, 32, 5, 32, 6, 32, 7, 32, 8, 32, 9, 32, 10, 32, 11, 32, 12, 32, 13, 32, 14, 32, 15};	
	      	  //auto maskAttr = rewriter.getI64ArrayAttr(mask);

		  auto maskAttr = mlir::DenseI64ArrayAttr::get(rewriter.getContext(), mask);
		  auto maskAttr2 = mlir::DenseI64ArrayAttr::get(rewriter.getContext(), mask2);


		  mlir::VectorType vecType = mlir::VectorType::get({32}, rewriter.getBF16Type());
		  llvm::APFloat zeroBF16 = llvm::APFloat::getZero(llvm::APFloat::BFloat());
		  SmallVector<llvm::APFloat, 32> bf16Values;
		  bf16Values.resize(32, zeroBF16);
		  mlir::DenseElementsAttr zeroAttr = mlir::DenseElementsAttr::get(vecType, bf16Values);
		  auto zeroVec = rewriter.create<mlir::arith::ConstantOp>(kForOp.getLoc(), vecType, zeroAttr);

    // Create vector.shuffle operation
    		 auto shuffledVector = rewriter.create<vector::ShuffleOp>(kForOp.getLoc(), bcast1, zeroVec, maskAttr);
		 auto shuffledVector2 = rewriter.create<vector::ShuffleOp>(kForOp.getLoc(), bcast2, shuffledVector, maskAttr2);


		  SmallVector<int64_t, 1> offsets_0 = {0};
		  SmallVector<int64_t, 1> offsets_16 = {16}; // Must be ≤ 16 if inserting vector<16xbf16> into vector<32xbf16>
		  SmallVector<int64_t, 1> strides = {1};

		  auto vect1 = rewriter.create<vector::InsertStridedSliceOp>(kForOp.getLoc(), bcast1, zeroVec, offsets_0, strides);
		  auto vect2 = rewriter.create<vector::InsertStridedSliceOp>(kForOp.getLoc(), bcast2, vect1, offsets_16, strides);

                  broadcasts.push_back(shuffledVector2);
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

		SmallVector<Value, 4> broadcasts_b;
		for (int i = 0, j = 0; i < vnni; i++, j = j + 16) {
		    Value indexOp = rewriter.create<arith::ConstantIndexOp>(reductionForOp.getLoc(), j);
		    auto rowVec = rewriterNewKForOp.create<vector::LoadOp>(
                    kForOp.getLoc(), VectorType::get({N}, elementType),
                    rhsClone->getResult(0), ValueRange{c0, c0,indexOp, c0});

		     broadcasts_b.push_back(rowVec);
		}

		
		mlir::VectorType dstType = mlir::VectorType::get({16}, rewriter.getF32Type());
		/*for (int i = 0, k = 0; i < vnni; i++, k = k + M) {
			for (int j = 0; j < M; j++) {
				auto dp = rewriter.create<mlir::x86vector::DotBF16Op>(
      				  kForOp.getLoc(), dstType, iterArgsNewKForOp[j+k], broadcasts[j], broadcasts_b[i]
   				 );
				bf16DP.push_back(dp);
			}
		}*/

		for (int i = 0, k = 0; i < M; i++, k = k + vnni) {
                        for (int j = 0; j < vnni; j++) {
                                auto dp = rewriter.create<mlir::x86vector::DotBF16Op>(
                                  kForOp.getLoc(), dstType, iterArgsNewKForOp[j+k], broadcasts[i], broadcasts_b[j]
                                 );
                                bf16DP.push_back(dp);
                        }
                }


                rewriterNewKForOp.create<scf::YieldOp>(locNewKForOp,
                                                       bf16DP);
              });
          rewriterNewReductionForOp.create<scf::YieldOp>(
              locNewReductionForOp, newKForOp.getResults());
        });

    VectorType resultType = VectorType::get({16}, rewriter.getBF16Type());

                for (int i = 0, k = 0; i < M; i++) {
			for (int j = 0; j < vnni; j++) {
				Value indexOp = rewriter.create<arith::ConstantIndexOp>(reductionForOp.getLoc(), j*16);
				Value bf16Vector = rewriter.create<x86vector::CvtPackedF32ToBF16Op>(
        reductionForOp.getLoc(), resultType, newReductionForOp.getResult(k));
				rewriter.create<vector::StoreOp>(reductionForOp.getLoc(), bf16Vector,
                                         subview_2_splits[i],
                                         ValueRange{c0, indexOp});

				k++;
			}
                }

    for (auto result : contractOp->getResults()) {
      for (auto *userOp : result.getUsers()) {
        rewriter.eraseOp(userOp);
      }
    }
    rewriter.eraseOp(contractOp);
    rewriter.eraseOp(vectorReadOpAcc);
    rewriter.eraseOp(vectorReadOpLhs);
    rewriter.eraseOp(vectorReadOpRhs);
    rewriter.eraseOp(subviewOpLhs);
    rewriter.eraseOp(subviewOpRhs);
    rewriter.eraseOp(kForOp);
    rewriter.eraseOp(reductionForOp);


    return success();
  }
};

void populateBF16DotProductPatterns(RewritePatternSet &patterns) {
  patterns.add<BF16DotProductOp>(patterns.getContext());
}

struct BF16DotProduct
    : public impl::BF16DotProductBase<BF16DotProduct> {
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
