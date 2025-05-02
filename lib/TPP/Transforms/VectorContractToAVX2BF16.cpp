//===- VectorContractToAVX2BF16.cpp -----------------------*- C++-*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of vector contraction using x86vector ops
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
#define GEN_PASS_DEF_AVX2BF16
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace mlir {
namespace tpp {

static FailureOr<SmallVector<scf::ForOp>>
getNestedLoop(vector::ContractionOp contractOp) {
  SmallVector<scf::ForOp> list;
  Operation *current = contractOp;
  // It is register tiled loop structure on batch reduce matmul
  // (M->N->Batch-reduce->K).
  // TODO: support for matmul and batch matmul
  for (int i = 0; i < 4; i++) {
    Operation *parent = current->getParentOfType<scf::ForOp>();
    if (!parent)
      return failure();
    list.push_back(dyn_cast<scf::ForOp>(parent));
    current = parent;
  }
  return list;
}

static bool isTransposedMatrix(vector::ContractionOp contractOp) {
  SmallVector<AffineMap, 3> contractMaps = contractOp.getIndexingMapsArray();
  AffineMap mapA = contractMaps[0];
  AffineMap mapB = contractMaps[1];

  auto resultsMapA = mapA.getNumResults();
  auto resultsMapB = mapB.getNumResults();
  assert(resultsMapA == 4 && resultsMapB == 4 &&
         "Result dim map for A and B should be 4");

  auto inputsMapA = mapA.getNumInputs();
  auto inputsMapB = mapB.getNumInputs();
  assert(inputsMapA == 5 && inputsMapB == 5 &&
         "Input dim map for A and B should be 5");

  auto vnniDim = dyn_cast<AffineDimExpr>(mapA.getResult(3));
  auto dimBR = dyn_cast<AffineDimExpr>(mapA.getResult(0));

  SmallVector<AffineDimExpr> listMxNxK;
  for (unsigned int i = 0; i < inputsMapA; i++) {
    auto affineExpr =
        dyn_cast<AffineDimExpr>(mlir::getAffineDimExpr(i, mapA.getContext()));
    if (affineExpr != vnniDim && affineExpr != dimBR)
      listMxNxK.push_back(affineExpr);
  }
  auto dimM = listMxNxK[0];
  auto dimN = listMxNxK[1];
  auto dimK = listMxNxK[2];

  // Transpose if the mapA is kxm
  if (dyn_cast<AffineDimExpr>(mapA.getResult(1)) == dimK &&
      dyn_cast<AffineDimExpr>(mapA.getResult(2)) == dimM)
    return true;
  // Transpose if the mapB is nxk
  if (dyn_cast<AffineDimExpr>(mapB.getResult(1)) == dimN &&
      dyn_cast<AffineDimExpr>(mapB.getResult(2)) == dimK)
    return true;

  return false;
}

static bool permutationCheck(vector::ContractionOp contractOp) {
  SmallVector<AffineMap, 3> contractMaps = contractOp.getIndexingMapsArray();
  AffineMap mapA = contractMaps[0];
  AffineMap mapB = contractMaps[1];

  auto inputsMapA = mapA.getNumInputs();
  SmallVector<AffineDimExpr> inputDims;
  for (unsigned int i = 0; i < inputsMapA; i++) {
    auto affineExpr =
        dyn_cast<AffineDimExpr>(mlir::getAffineDimExpr(i, mapA.getContext()));
    inputDims.push_back(affineExpr);
  }

  bool flag = true;
  // mapA result dims
  auto resultsMapA = mapA.getNumResults();
  SmallVector<AffineDimExpr> outputDimsA;
  for (unsigned int i = 0; i < resultsMapA; i++) {
    auto affineExpr = dyn_cast<AffineDimExpr>(mapA.getResult(i));
    outputDimsA.push_back(affineExpr);
  }

  // We match the pattern {Batch-reduction, vnni, M, N, K} or {Batch-reduction,
  // M, N, K, vnni} -> {Batch-reduction, M, K, vnni}
  auto c1 = inputDims[0] == outputDimsA[0];
  auto c2 = (inputDims[1] == outputDimsA[3]) &&
            (inputDims[2] == outputDimsA[1]) &&
            (inputDims[4] == outputDimsA[2]);
  auto c3 = (inputDims[1] == outputDimsA[1]) &&
            (inputDims[3] == outputDimsA[2]) &&
            (inputDims[4] == outputDimsA[3]);
  flag = flag && (c1 && (c2 || c3));

  // mapB result dims
  auto resultsMapB = mapB.getNumResults();
  SmallVector<AffineDimExpr> outputDimsB;
  for (unsigned int i = 0; i < resultsMapB; i++) {
    auto affineExpr = dyn_cast<AffineDimExpr>(mapB.getResult(i));
    outputDimsB.push_back(affineExpr);
  }

  // We match the pattern {Batch-reduction, vnni, M, N, K} or {Batch-reduction,
  // M, N, K, vnni} -> {Batch-reduction, K, N, vnni}
  auto c4 = inputDims[0] == outputDimsB[0];
  auto c5 = (inputDims[1] == outputDimsB[3]) &&
            (inputDims[4] == outputDimsB[1]) &&
            (inputDims[3] == outputDimsB[2]);
  auto c6 = (inputDims[2] == outputDimsB[2]) &&
            (inputDims[3] == outputDimsB[1]) &&
            (inputDims[4] == outputDimsB[3]);
  flag = flag && (c4 && (c5 || c6));

  return flag;
}

/// This pass lowers vector.contract (linalg.batch_reduce_matmul) for bf16
/// (vnni=2) type into sequence of x86vector::DotBF16Op.
///
/// As an example, the following pseudo-code will be rewritten
/// scf.for // m-tile
///  scf.for // n-tile
///   subview // C matrix
///   scf.for // batch-reduce
///   scf.for // k-tile
///    subview // A and B matrix
///    vector.read // A, B, and C matrix
///    vector.contract
///    vector.write // to C matrix
///
/// to:
///
/// scf.for // m-tile
///  scf.for // n-tile
///   vector.load // load C matrix in chunks of <16xbf16>
///   arith.shli + vector.bitcast // upconvert to f32 and pass them as iterargs
///   scf.for (iterargs = C matrix load as f32) // batch-reduce
///    scf.for (iterargs = batch-reduce iterArgs) // k-tile
///     vector.load // load 2 elements of A matrix and broadcast them into
///     <32xbf16> vector.load // load elements of B matrix into <32xbf16>
///     x86vector.avx512.dot %iterargs, %Ax, %Bx // accumulate in f32 (via
///     iterargs) x86vector.avx512.dot %iterargs, %Ax, %By // accumulate in f32
///     (via iterargs)
///     ..............
///     ..............
///    scf.yield // yield dpbf16 results
///   scf.yield // yield results of scf.for k-tile
///  arith.truncate // downconvert accumulator value from f32 to bf16
///  vector.store // store back into C matrix
///  .............
///  ............

struct AVX2BF16Op : OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {

    if (contractOp.getKind() != vector::CombiningKind::ADD)
      return rewriter.notifyMatchFailure(
          contractOp, "Only combining kind 'ADD' is supported now.");

    // Check the vector contract operation satisfies the required pattern.
    // Check the Acc, Lhs, and Rhs of contract operation

    auto loops = getNestedLoop(contractOp);
    if (failed(loops))
      return rewriter.notifyMatchFailure(
          contractOp, "Invalid loop nest in contract pattern");

    auto nestedLoops = *loops;
    auto kForOp = nestedLoops[0];
    auto reductionForOp = nestedLoops[1];

    auto vectorReadOpAcc =
        reductionForOp.getInitArgs()[0].getDefiningOp<vector::TransferReadOp>();
    auto vectorReadOpLhs =
        contractOp.getLhs().getDefiningOp<vector::TransferReadOp>();
    auto vectorReadOpRhs =
        contractOp.getRhs().getDefiningOp<vector::TransferReadOp>();

    // Check whether the operand of vector transfer read is a subview
    auto subviewOpAcc =
        vectorReadOpAcc.getOperand(0).getDefiningOp<memref::SubViewOp>();
    auto elementType =
        (cast<MemRefType>(subviewOpAcc.getType())).getElementType();

    if (!elementType.isBF16())
      return rewriter.notifyMatchFailure(contractOp, "The type is not BF16");

    // Check the operation type MatMul, B-MatMul, or BR-MatMul (FP32/BF16)
    SmallVector<vector::IteratorType> contractIteratorTypes =
        contractOp.getIteratorTypesArray();
    int reductionCount =
        std::count(contractIteratorTypes.begin(), contractIteratorTypes.end(),
                   vector::IteratorType::reduction);

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

    if (reductionCount == 3 &&
        (lhsType.getRank() != 4 || rhsType.getRank() != 4))
      return rewriter.notifyMatchFailure(
          contractOp,
          "Invalid rank for batch reduce operation with vnni layout");

    int64_t M = lhsType.getDimSize(lhsType.getRank() - 3);
    int64_t N = rhsType.getDimSize(lhsType.getRank() - 2);
    int64_t K = lhsType.getDimSize(lhsType.getRank() - 2);
    int64_t vnni = lhsType.getDimSize(lhsType.getRank() - 1);

    // K tile should be equal to vnni layout
    if (K != (vnni / 2))
      return rewriter.notifyMatchFailure(
          contractOp, "K tile size should be equal to VNNI layout");

    if ((N % 8) != 0)
      return rewriter.notifyMatchFailure(
          contractOp, "N tile size divisible by 8 are only supported");

    if (vnni != 2)
      return rewriter.notifyMatchFailure(
          contractOp, "Only VNNI layout=2 is supported, now");

    if (isTransposedMatrix(contractOp))
      return rewriter.notifyMatchFailure(contractOp,
                                         "Matrices shoudn't be transposed.");

    if (!permutationCheck(contractOp))
      return rewriter.notifyMatchFailure(
          contractOp, "Affine map permutation not supported.");

    rewriter.setInsertionPoint(reductionForOp);

    // Creating further subviews from the C matrix subview
    llvm::SmallVector<Value, 8> subviewCMatrix;
    llvm::SmallVector<Value, 8> loopItrArgs;
    for (int j = 0; j < N; j = j + 8) {
      for (int i = 0; i < M; i++) {
        Value indexOp_A =
            rewriter.create<arith::ConstantIndexOp>(reductionForOp.getLoc(), i);
        Value indexOp_B =
            rewriter.create<arith::ConstantIndexOp>(reductionForOp.getLoc(), j);
        auto valueCRow = rewriter.create<vector::LoadOp>(
            reductionForOp.getLoc(), VectorType::get(8, elementType),
            subviewOpAcc, ValueRange{indexOp_A, indexOp_B});
        auto bitcast_i16 = rewriter.create<vector::BitCastOp>(
            reductionForOp.getLoc(),
            VectorType::get(8, rewriter.getIntegerType(16)), valueCRow);
        auto extend_i32 = rewriter.create<arith::ExtUIOp>(
            reductionForOp.getLoc(),
            VectorType::get(8, rewriter.getIntegerType(32)), bitcast_i16);
        auto cst16 = rewriter.create<arith::ConstantOp>(
            reductionForOp.getLoc(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(32), 16));
        auto vectType = VectorType::get(8, rewriter.getIntegerType(32));
        auto shiftOp = rewriter.create<arith::ShLIOp>(
            reductionForOp.getLoc(), vectType, extend_i32,
            rewriter.create<vector::BroadcastOp>(reductionForOp.getLoc(),
                                                 vectType, cst16));
        auto f32CVector = rewriter.create<vector::BitCastOp>(
            reductionForOp.getLoc(), VectorType::get(8, rewriter.getF32Type()),
            shiftOp);
        loopItrArgs.push_back(f32CVector);
      }
    }

    SmallVector<Value, 8> bf16FMAs;

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

                IRMapping rhsMapping;
                rhsMapping.map(
                    vectorReadOpRhs.getSource().getDefiningOp()->getOperand(1),
                    ivNewReductionForOp);
                rhsMapping.map(
                    vectorReadOpRhs.getSource().getDefiningOp()->getOperand(2),
                    ivNewKForOp);
                auto rhsClone = rewriterNewKForOp.clone(
                    *vectorReadOpRhs.getSource().getDefiningOp(), rhsMapping);

                for (int j = 0, k = 0; j < N; j = j + 8) {

		  // Changes for A
		  //
  		auto attr = rewriter.getI32IntegerAttr(-65536);
  		auto maskConst = rewriter.create<mlir::arith::ConstantOp>(kForOp.getLoc(), rewriter.getI32Type(), attr);
		auto maskBcst = rewriterNewKForOp.create<vector::BroadcastOp>(
                          kForOp.getLoc(),
                          VectorType::get(8, rewriterNewKForOp.getI32Type()),
                          maskConst);
	        
		Value indexOp_c0 =
            		rewriter.create<arith::ConstantIndexOp>(reductionForOp.getLoc(), 0);
        	auto valueARow = rewriter.create<vector::LoadOp>(
            		kForOp.getLoc(), VectorType::get(2, elementType),
            		lhsClone->getResult(0), ValueRange{indexOp_c0, indexOp_c0, indexOp_c0, indexOp_c0});
		auto bitcast_i32 = rewriter.create<vector::BitCastOp>(
                 	kForOp.getLoc(),
            		VectorType::get(1, rewriter.getIntegerType(32)), valueARow);
		auto broadcastValueA =
                      rewriterNewKForOp.create<vector::BroadcastOp>(
                          kForOp.getLoc(),
                          VectorType::get(8, rewriterNewKForOp.getI32Type()),
                          bitcast_i32);
		auto andOp = rewriter.create<arith::AndIOp>(kForOp.getLoc(), broadcastValueA, maskBcst);
		auto AOdd = rewriter.create<vector::BitCastOp>(
                        kForOp.getLoc(),
                        VectorType::get(8, rewriter.getF32Type()), andOp);

		//changes for b
		Value indexOp_cj =
                        rewriter.create<arith::ConstantIndexOp>(reductionForOp.getLoc(), j);
		auto valueBRow = rewriter.create<vector::LoadOp>(
                        kForOp.getLoc(), VectorType::get(16, elementType),
                        rhsClone->getResult(0), ValueRange{indexOp_c0, indexOp_c0, indexOp_cj, indexOp_c0});
		auto bitcast_i32_B = rewriter.create<vector::BitCastOp>(
                        kForOp.getLoc(),
                        VectorType::get(8, rewriter.getIntegerType(32)), valueBRow);
		auto andOpB = rewriter.create<arith::AndIOp>(kForOp.getLoc(), bitcast_i32_B, maskBcst);
		auto BOdd = rewriter.create<vector::BitCastOp>(
                        kForOp.getLoc(),
                        VectorType::get(8, rewriter.getF32Type()), andOpB);

		//fma
                auto fmaOdd = rewriter.create<vector::FMAOp>(
                      kForOp.getLoc(), AOdd, BOdd, iterArgsNewKForOp[k]);
		k++;

		auto cst16 = rewriter.create<arith::ConstantOp>(
            		kForOp.getLoc(),
            		rewriter.getIntegerAttr(rewriter.getIntegerType(32), 16));
		auto vectType = VectorType::get(8, rewriter.getIntegerType(32));
        	auto shiftOpA = rewriter.create<arith::ShLIOp>(
            		kForOp.getLoc(), vectType, broadcastValueA,
           	 	rewriter.create<vector::BroadcastOp>(kForOp.getLoc(),
                                                 vectType, cst16));
		auto AEven = rewriter.create<vector::BitCastOp>(
                        kForOp.getLoc(),
                        VectorType::get(8, rewriter.getF32Type()), shiftOpA);

		auto shiftOpB = rewriter.create<arith::ShLIOp>(
                        kForOp.getLoc(), vectType, bitcast_i32_B,
                        rewriter.create<vector::BroadcastOp>(kForOp.getLoc(),
                                                 vectType, cst16));
		auto BEven = rewriter.create<vector::BitCastOp>(
                        kForOp.getLoc(),
                        VectorType::get(8, rewriter.getF32Type()), shiftOpB);

	   	//fma
                auto fmaEven = rewriter.create<vector::FMAOp>(
                      kForOp.getLoc(), AEven, BEven, fmaOdd);



                  bf16FMAs.push_back(fmaEven);

                  for (int i = 1; i < M; i++) {



                Value indexOp_c0 =
                        rewriter.create<arith::ConstantIndexOp>(reductionForOp.getLoc(), 0);
		Value indexOp_ci =
                        rewriter.create<arith::ConstantIndexOp>(reductionForOp.getLoc(), i);
                auto valueA1Row = rewriter.create<vector::LoadOp>(
                        kForOp.getLoc(), VectorType::get(2, elementType),
                        lhsClone->getResult(0), ValueRange{indexOp_c0, indexOp_ci, indexOp_c0, indexOp_c0});
                auto bitcast_i32_A1 = rewriter.create<vector::BitCastOp>(
                        kForOp.getLoc(),
                        VectorType::get(1, rewriter.getIntegerType(32)), valueA1Row);
                auto broadcastValueA1 =
                      rewriterNewKForOp.create<vector::BroadcastOp>(
                          kForOp.getLoc(),
                          VectorType::get(8, rewriterNewKForOp.getI32Type()),
                          bitcast_i32_A1);
                auto andOpA1 = rewriter.create<arith::AndIOp>(kForOp.getLoc(), broadcastValueA1, maskBcst);
                auto A1Odd = rewriter.create<vector::BitCastOp>(
                        kForOp.getLoc(),
                        VectorType::get(8, rewriter.getF32Type()), andOpA1);


                    auto fmaOdd_m = rewriter.create<vector::FMAOp>(
                        kForOp.getLoc(), A1Odd, BOdd, iterArgsNewKForOp[k]);
                    k++;



                auto shiftOpA1 = rewriter.create<arith::ShLIOp>(
                        kForOp.getLoc(), vectType, broadcastValueA1,
                        rewriter.create<vector::BroadcastOp>(kForOp.getLoc(),
                                                 vectType, cst16));
                auto A1Even = rewriter.create<vector::BitCastOp>(
                        kForOp.getLoc(),
                        VectorType::get(8, rewriter.getF32Type()), shiftOpA1);



                    auto fmaEven_m = rewriter.create<vector::FMAOp>(
                        kForOp.getLoc(), A1Even, BEven, fmaOdd_m);

                    bf16FMAs.push_back(fmaEven_m);
                  }
                }
                rewriterNewKForOp.create<scf::YieldOp>(locNewKForOp, bf16FMAs);
              });

          rewriterNewReductionForOp.create<scf::YieldOp>(
              locNewReductionForOp, newKForOp.getResults());
        });

    for (int j = 0, k = 0; j < N; j = j + 8) {
      for (int i = 0; i < M; i++) {
        Value indexOp =
            rewriter.create<arith::ConstantIndexOp>(reductionForOp.getLoc(), i);
        Value indexOp_B =
            rewriter.create<arith::ConstantIndexOp>(reductionForOp.getLoc(), j);
        auto bf16vec = rewriter.create<arith::TruncFOp>(
            reductionForOp.getLoc(),
            VectorType::get({8}, rewriter.getBF16Type()),
            newReductionForOp.getResult(k));
        rewriter.create<vector::StoreOp>(reductionForOp.getLoc(), bf16vec,
                                         subviewOpAcc,
                                         ValueRange{indexOp, indexOp_B});
        k++;
      }
    }

    Value contractVal = reductionForOp.getResult(0);
    Operation *writeOp;
    for (auto val : contractVal.getUsers()) {
      writeOp = dyn_cast<vector::TransferWriteOp>(val);
      if (writeOp) {
        rewriter.eraseOp(writeOp);
        break;
      }
    }

    return success();
  }
};

void populateAVX2BF16Patterns(RewritePatternSet &patterns) {
  patterns.add<AVX2BF16Op>(patterns.getContext());
}

struct AVX2BF16 : public impl::AVX2BF16Base<AVX2BF16> {
  using AVX2BF16Base::AVX2BF16Base;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateAVX2BF16Patterns(patterns);
    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;
    (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
  }
};
} // namespace tpp
} // namespace mlir
