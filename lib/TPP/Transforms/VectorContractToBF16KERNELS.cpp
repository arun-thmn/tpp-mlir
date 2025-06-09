//===- VectorContractToBF16KERNELS.cpp -----------------------*- C++-*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of vector contraction using x86vector ops
//
//===----------------------------------------------------------------------===//
#include "TPP/Transforms/Utils/VNNIUtils.h"
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
#define GEN_PASS_DEF_BF16KERNELS
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

struct BF16KERNELSOp : OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {

    // We get target architecture and decide on uKernel lowering using flags
    auto cpuName = vnni::utils::getTargetArchName();
    bool adl = false;
    bool srf = false;
    bool clx = false;
    bool cpx = false;

    int sizeFactor = 0;
    if (cpuName == "SRF") {
      sizeFactor = 8;
      srf = true;
    }

    if (cpuName == "ADL") {
      sizeFactor = 8;
      adl = true;
    }

    if (cpuName == "CLX") {
      sizeFactor = 16;
      clx = true;
    }

    if (cpuName == "CPX") {
      sizeFactor = 16;
      cpx = true;
    }

    if (!(srf || adl || clx || cpx))
      return rewriter.notifyMatchFailure(
          contractOp, "This pass supports uKernel lowering only for "
                      "SRF/ADL/ARL/CLX/CPX kind of machines");

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
    auto nForOp = nestedLoops[2];
    auto mForOp = nestedLoops[3];

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

    if (srf) {
    if (!(elementType.isBF16() || elementType.isF16()))
      return rewriter.notifyMatchFailure(contractOp,
                                         "The type is not BF16 or F16");
    } else {
    if (!(elementType.isBF16()))
      return rewriter.notifyMatchFailure(contractOp,
                                         "The type is not BF16");
    }

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

    if ((N % 16) != 0)
      return rewriter.notifyMatchFailure(
          contractOp, "N tile size divisible by 16 are only supported");

    if (vnni != 2)
      return rewriter.notifyMatchFailure(
          contractOp, "Only VNNI layout=2 is supported, now");

    if (isTransposedMatrix(contractOp))
      return rewriter.notifyMatchFailure(contractOp,
                                         "Matrices shoudn't be transposed.");

    if (!permutationCheck(contractOp))
      return rewriter.notifyMatchFailure(
          contractOp, "Affine map permutation not supported.");

    rewriter.setInsertionPoint(mForOp);
    auto i32Type = rewriter.getIntegerType(32);
    auto i16Type = rewriter.getIntegerType(16);
    Value indexOp_c0 =
        rewriter.create<arith::ConstantIndexOp>(reductionForOp.getLoc(), 0);
    auto cst16 = rewriter.create<arith::ConstantOp>(
        reductionForOp.getLoc(), rewriter.getIntegerAttr(i32Type, 16));

    // Creating the mask for doing bitwise and operation (target: ADL)
    auto intAttr = rewriter.getI32IntegerAttr(-65536);
    auto maskConst = rewriter.create<mlir::arith::ConstantOp>(
        reductionForOp.getLoc(), i32Type, intAttr);
    auto mBcst = rewriter.create<vector::BroadcastOp>(
        reductionForOp.getLoc(), VectorType::get(sizeFactor, i32Type),
        maskConst);
    auto memrefMask = rewriter.create<mlir::memref::AllocOp>(
        reductionForOp.getLoc(),
        MemRefType::get({1}, VectorType::get(sizeFactor, i32Type)));
    rewriter.create<memref::StoreOp>(reductionForOp.getLoc(), mBcst, memrefMask,
                                     ValueRange{indexOp_c0});

    rewriter.setInsertionPoint(reductionForOp);
    llvm::SmallVector<Value, 8> loopItrArgs;

    // C Matrix load + upconvert to f32
    if (elementType.isBF16()) {
      for (int j = 0; j < N; j = j + sizeFactor) {
        for (int i = 0; i < M; i++) {
          Value indexOp_A = rewriter.create<arith::ConstantIndexOp>(
              reductionForOp.getLoc(), i);
          Value indexOp_B = rewriter.create<arith::ConstantIndexOp>(
              reductionForOp.getLoc(), j);
          auto valueCRow = rewriter.create<vector::LoadOp>(
              reductionForOp.getLoc(), VectorType::get(sizeFactor, elementType),
              subviewOpAcc, ValueRange{indexOp_A, indexOp_B});
          auto bitcast_i16 = rewriter.create<vector::BitCastOp>(
              reductionForOp.getLoc(), VectorType::get(sizeFactor, i16Type),
              valueCRow);
          auto extend_i32 = rewriter.create<arith::ExtUIOp>(
              reductionForOp.getLoc(), VectorType::get(sizeFactor, i32Type),
              bitcast_i16);
          auto vectType = VectorType::get(sizeFactor, i32Type);
          auto shiftOp = rewriter.create<arith::ShLIOp>(
              reductionForOp.getLoc(), vectType, extend_i32,
              rewriter.create<vector::BroadcastOp>(reductionForOp.getLoc(),
                                                   vectType, cst16));
          auto f32CVector = rewriter.create<vector::BitCastOp>(
              reductionForOp.getLoc(),
              VectorType::get(sizeFactor, rewriter.getF32Type()), shiftOp);
          loopItrArgs.push_back(f32CVector);
        }
      }
    }

    if (elementType.isF16()) {
      for (int j = 0; j < N; j = j + sizeFactor) {
        for (int i = 0; i < M; i++) {
          Value indexOp_A = rewriter.create<arith::ConstantIndexOp>(
              reductionForOp.getLoc(), i);
          Value indexOp_B = rewriter.create<arith::ConstantIndexOp>(
              reductionForOp.getLoc(), j);
          auto valueCRow = rewriter.create<vector::LoadOp>(
              reductionForOp.getLoc(), VectorType::get(sizeFactor, elementType),
              subviewOpAcc, ValueRange{indexOp_A, indexOp_B});
          auto f32CVector = rewriter.create<arith::ExtFOp>(
              reductionForOp.getLoc(),
              VectorType::get({8}, rewriter.getF32Type()),
              valueCRow.getResult());
          loopItrArgs.push_back(f32CVector);
        }
      }
    }

    SmallVector<Value, 8> evenFMAs;
    SmallVector<Value, 8> oddFMAs;
    SmallVector<Value, 8> matf32;

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
                    vectorReadOpLhs.getBase().getDefiningOp()->getOperand(1),
                    ivNewReductionForOp);
                mapping.map(
                    vectorReadOpLhs.getBase().getDefiningOp()->getOperand(3),
                    ivNewKForOp);
                auto lhsClone = rewriterNewKForOp.clone(
                    *vectorReadOpLhs.getBase().getDefiningOp(), mapping);

                IRMapping rhsMapping;
                rhsMapping.map(
                    vectorReadOpRhs.getBase().getDefiningOp()->getOperand(1),
                    ivNewReductionForOp);
                rhsMapping.map(
                    vectorReadOpRhs.getBase().getDefiningOp()->getOperand(2),
                    ivNewKForOp);
                auto rhsClone = rewriterNewKForOp.clone(
                    *vectorReadOpRhs.getBase().getDefiningOp(), rhsMapping);

                // Load i32 mask that is created and stored in memory earlier
                auto maskBcst = rewriter.create<memref::LoadOp>(
                    kForOp.getLoc(), VectorType::get(sizeFactor, i32Type),
                    memrefMask, ValueRange{indexOp_c0});

                // i1 type mask
                auto boolAttr_16 = DenseElementsAttr::get(
                    VectorType::get({16}, rewriter.getI1Type()),
                    ArrayRef<bool>{false, true, false, true, false, true, false,
                                   true, false, true, false, true, false, true,
                                   false, true});
                auto i1Mask_16 = rewriter.create<arith::ConstantOp>(
                    kForOp.getLoc(),
                    VectorType::get({16}, rewriter.getI1Type()), boolAttr_16);
                auto boolAttr_2 = DenseElementsAttr::get(
                    VectorType::get({vnni}, rewriter.getI1Type()),
                    ArrayRef<bool>{false, true});
                auto i1Mask_2 = rewriter.create<arith::ConstantOp>(
                    kForOp.getLoc(),
                    VectorType::get(vnni, rewriter.getI1Type()), boolAttr_2);

                auto zeroAttr = rewriter.getFloatAttr(elementType, 0.0);

                // Destination type
                mlir::VectorType dstType =
                    mlir::VectorType::get(sizeFactor, rewriter.getF32Type());

                llvm::SmallVector<OpFoldResult> strides = {
                    rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
                    rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};
                llvm::SmallVector<OpFoldResult> sizes = {
                    rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
                    rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};

                // uKernel lowering for machines like zen5 to target
                // avx512bf16dp
                if (cpx) {
                  // Load elements of B matrix and store in a DS
                  for (int j = 0; j < N; j = j + sizeFactor) {
                    Value indexOp_j = rewriter.create<arith::ConstantIndexOp>(
                        reductionForOp.getLoc(), j);
                    auto valueRow = rewriterNewKForOp.create<vector::LoadOp>(
                        kForOp.getLoc(), VectorType::get(32, elementType),
                        rhsClone->getResult(0),
                        ValueRange{indexOp_c0, indexOp_c0, indexOp_j,
                                   indexOp_c0});
                    matf32.push_back(valueRow);
                  }

                  // Load elements of A matrix, do FMA, and store in a DS
                  for (int i = 0; i < M; i++) {
                    Value indexOp_i = rewriter.create<arith::ConstantIndexOp>(
                        reductionForOp.getLoc(), i);
                    auto valueRow = rewriterNewKForOp.create<vector::LoadOp>(
                        kForOp.getLoc(), VectorType::get({vnni}, elementType),
                        lhsClone->getResult(0),
                        ValueRange{indexOp_c0, indexOp_i, indexOp_c0,
                                   indexOp_c0});
                    auto bitcastValue_i32 =
                        rewriterNewKForOp.create<vector::BitCastOp>(
                            kForOp.getLoc(),
                            VectorType::get({1},
                                            rewriterNewKForOp.getI32Type()),
                            valueRow);
                    auto bcst_i32 =
                        rewriterNewKForOp.create<vector::BroadcastOp>(
                            kForOp.getLoc(),
                            VectorType::get(sizeFactor,
                                            rewriterNewKForOp.getI32Type()),
                            bitcastValue_i32);
                    auto valuef32 = rewriterNewKForOp.create<vector::BitCastOp>(
                        kForOp.getLoc(),
                        VectorType::get(32, rewriterNewKForOp.getBF16Type()),
                        bcst_i32);
                    for (int j = 0; j < (N / sizeFactor); j++) {
                      auto dp = rewriter.create<mlir::x86vector::DotBF16Op>(
                          kForOp.getLoc(), dstType,
                          iterArgsNewKForOp[i + (j * M)], valuef32, matf32[j]);
                      oddFMAs.push_back(dp);
                    }
                  }

                  // Re-arrange the stored FMAs in order
                  for (int j = 0; j < (N / sizeFactor); j++) {
                    for (int i = 0; i < M; i++) {
                      evenFMAs.push_back(oddFMAs[j + (i * (N / sizeFactor))]);
                    }
                  }
                }

		// uKernel lowering for machines that has no support for avx512bf16dp and AMX
                if (clx) {
		  // Load odd elements of B Matrix and store in a DS
                  for (int j = 0; j < N; j = j + sizeFactor) {
                    Value indexOp_cj = rewriter.create<arith::ConstantIndexOp>(
                        reductionForOp.getLoc(), j);
                    auto valueRow = rewriter.create<vector::LoadOp>(
                        kForOp.getLoc(), VectorType::get(32, elementType),
                        rhsClone->getResult(0),
                        ValueRange{indexOp_c0, indexOp_c0, indexOp_cj,
                                   indexOp_c0});
                    auto bitcast_i32 = rewriter.create<vector::BitCastOp>(
                        kForOp.getLoc(), VectorType::get(sizeFactor, i32Type),
                        valueRow);
                    auto andOpB = rewriter.create<arith::AndIOp>(
                        kForOp.getLoc(), bitcast_i32, maskBcst);
                    auto oddB = rewriter.create<vector::BitCastOp>(
                        kForOp.getLoc(),
                        VectorType::get(sizeFactor, rewriter.getF32Type()),
                        andOpB);
                    matf32.push_back(oddB);
                  }

		  // Load odd elements of A Matrix, perform fma (odd), and store
                  // to a DS
                  for (int i = 0; i < M; i++) {
                    Value indexOp_ci = rewriter.create<arith::ConstantIndexOp>(
                        reductionForOp.getLoc(), i);
                    auto denseAttr = DenseElementsAttr::get(
                        VectorType::get(vnni, elementType), zeroAttr);
                    auto passThru = rewriter.create<arith::ConstantOp>(
                        kForOp.getLoc(), VectorType::get(vnni, elementType),
                        denseAttr);
                    auto maskedLoad = rewriter.create<vector::MaskedLoadOp>(
                        kForOp.getLoc(), VectorType::get(vnni, elementType),
                        lhsClone->getResult(0),
                        ValueRange{indexOp_c0, indexOp_ci, indexOp_c0,
                                   indexOp_c0},
                        i1Mask_2, passThru);
                    auto bitcast_f32 = rewriter.create<vector::BitCastOp>(
                        kForOp.getLoc(),
                        VectorType::get(1, rewriter.getF32Type()), maskedLoad);
                    auto oddA = rewriterNewKForOp.create<vector::BroadcastOp>(
                        kForOp.getLoc(),
                        VectorType::get(sizeFactor,
                        rewriterNewKForOp.getF32Type()), bitcast_f32);

		    // Odd FMAs
                    for (int j = 0; j < (N / sizeFactor); j++) {
                      auto fmaOdd = rewriter.create<vector::FMAOp>(
                          kForOp.getLoc(), oddA, matf32[j],
                          iterArgsNewKForOp[i + (j * M)]);
                      oddFMAs.push_back(fmaOdd);
                    }
                  }

		  // Clear B matrix odd loads to save even loads
                  matf32.clear();

		  // Load even elements of B Matrix and store in a DS
                  for (int j = 0; j < N; j = j + sizeFactor) {
                    Value indexOp_cj = rewriter.create<arith::ConstantIndexOp>(
                        reductionForOp.getLoc(), j);
                    auto valueRow = rewriter.create<vector::LoadOp>(
                        kForOp.getLoc(), VectorType::get(32, elementType),
                        rhsClone->getResult(0),
                        ValueRange{indexOp_c0, indexOp_c0, indexOp_cj,
                                   indexOp_c0});
                    auto bitcast_i32 = rewriter.create<vector::BitCastOp>(
                        kForOp.getLoc(), VectorType::get(sizeFactor, i32Type),
                        valueRow);
                    auto shiftOpB = rewriter.create<arith::ShLIOp>(
                        kForOp.getLoc(), VectorType::get(sizeFactor, i32Type),
                        bitcast_i32,
                        rewriter.create<vector::BroadcastOp>(
                            kForOp.getLoc(),
                            VectorType::get(sizeFactor, i32Type), cst16));
                    auto evenB = rewriter.create<vector::BitCastOp>(
                        kForOp.getLoc(),
                        VectorType::get(sizeFactor, rewriter.getF32Type()),
                        shiftOpB);
                    matf32.push_back(evenB);
                  }

                  SmallVector<Value, 8> evenFMAs_swap;
		  // Load even elements of A Matrix, perform fma (even), and store
                  // to a DS
                  for (int i = 0, k = 0; i < M; i++) {
                    Value indexOp_ci = rewriter.create<arith::ConstantIndexOp>(
                        reductionForOp.getLoc(), i);
                    auto valueRow = rewriter.create<vector::LoadOp>(
                        kForOp.getLoc(), VectorType::get(vnni, elementType),
                        lhsClone->getResult(0),
                        ValueRange{indexOp_c0, indexOp_ci, indexOp_c0,
                                   indexOp_c0});
                    auto bitcast_i32 = rewriter.create<vector::BitCastOp>(
                        kForOp.getLoc(), VectorType::get(1, i32Type), valueRow);
                    auto bcstValue =
                        rewriterNewKForOp.create<vector::BroadcastOp>(
                            kForOp.getLoc(),
                            VectorType::get(sizeFactor, i32Type), bitcast_i32);
                    auto shiftOp = rewriter.create<arith::ShLIOp>(
                        kForOp.getLoc(), VectorType::get(sizeFactor, i32Type),
                        bcstValue,
                        rewriter.create<vector::BroadcastOp>(
                            kForOp.getLoc(),
                            VectorType::get(sizeFactor, i32Type), cst16));
                    auto evenA = rewriter.create<vector::BitCastOp>(
                        kForOp.getLoc(),
                        VectorType::get(sizeFactor, rewriter.getF32Type()),
                        shiftOp);

		    // Even FMAs
                    for (int j = 0; j < (N / sizeFactor); j++) {
                      auto fmaEven = rewriter.create<vector::FMAOp>(
                          kForOp.getLoc(), evenA, matf32[j], oddFMAs[k]);
                      evenFMAs_swap.push_back(fmaEven);
                      k++;
                    }
                  }

		  //  Re-arrange the stored FMAs in order
                  for (int j = 0; j < (N / sizeFactor); j++) {
                    for (int i = 0; i < M; i++) {
                      evenFMAs.push_back(
                          evenFMAs_swap[j + (i * (N / sizeFactor))]);
                    }
                  }
                } 

		// uKernel lowering for AVX2  machines
                if (srf || adl) {
                  // Load odd elements of A Matrix and store in a DS
                  for (int i = 0; i < M; i++) {
                    Value oddA;

                    if (adl) {
                      Value indexOp_ci =
                          rewriter.create<arith::ConstantIndexOp>(
                              reductionForOp.getLoc(), i);
                      auto valueRow = rewriter.create<vector::LoadOp>(
                          kForOp.getLoc(), VectorType::get(2, elementType),
                          lhsClone->getResult(0),
                          ValueRange{indexOp_c0, indexOp_ci, indexOp_c0,
                                     indexOp_c0});
                      auto bitcast_i32 = rewriter.create<vector::BitCastOp>(
                          kForOp.getLoc(), VectorType::get(1, i32Type),
                          valueRow);
                      auto bcst_i32 =
                          rewriterNewKForOp.create<vector::BroadcastOp>(
                              kForOp.getLoc(),
                              VectorType::get(sizeFactor, i32Type),
                              bitcast_i32);
                      auto andOpA = rewriter.create<arith::AndIOp>(
                          kForOp.getLoc(), bcst_i32, maskBcst);
                      oddA = rewriter.create<vector::BitCastOp>(
                          kForOp.getLoc(),
                          VectorType::get(sizeFactor, rewriter.getF32Type()),
                          andOpA);
                    }

                    if (srf) {
                      SmallVector<OpFoldResult> offsets = {
                          rewriter.getIndexAttr(0),
                          rewriter.getIndexAttr(i),
                          rewriter.getIndexAttr(0),
                          rewriter.getIndexAttr(1),
                      };
                      auto subview = rewriter.create<memref::SubViewOp>(
                          kForOp.getLoc(), lhsClone->getResult(0), offsets,
                          sizes, strides);
                      oddA =
                          rewriter.create<mlir::x86vector::BcstToPackedF32Op>(
                              kForOp.getLoc(), dstType, subview);
                    }

                    matf32.push_back(oddA);
                  }

                  // Load odd elements of B-Matrix, perform fma (odd), and store
                  // to a DS
                  for (int j = 0, k = 0; j < N; j = j + sizeFactor) {
                    Value oddB;

                    if (adl) {
                      Value indexOp_cj =
                          rewriter.create<arith::ConstantIndexOp>(
                              reductionForOp.getLoc(), j);
                      auto valueRow = rewriter.create<vector::LoadOp>(
                          kForOp.getLoc(), VectorType::get(16, elementType),
                          rhsClone->getResult(0),
                          ValueRange{indexOp_c0, indexOp_c0, indexOp_cj,
                                     indexOp_c0});
                      auto bitcast_i16 = rewriter.create<vector::BitCastOp>(
                          kForOp.getLoc(), VectorType::get(16, i16Type),
                          valueRow);
                      auto bitcast_i16_andOpA =
                          rewriter.create<vector::BitCastOp>(
                              kForOp.getLoc(), VectorType::get(16, i16Type),
                              matf32[0]);
                      auto selectOp = rewriter.create<arith::SelectOp>(
                          kForOp.getLoc(), VectorType::get({16}, i16Type),
                          i1Mask_16, bitcast_i16, bitcast_i16_andOpA);
                      oddB = rewriter.create<vector::BitCastOp>(
                          kForOp.getLoc(),
                          VectorType::get(sizeFactor, rewriter.getF32Type()),
                          selectOp);
                    }

                    if (srf) {
                      SmallVector<OpFoldResult> offsets = {
                          rewriter.getIndexAttr(0),
                          rewriter.getIndexAttr(0),
                          rewriter.getIndexAttr(j),
                          rewriter.getIndexAttr(0),
                      };
                      llvm::SmallVector<OpFoldResult> sizes = {
                          rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
                          rewriter.getIndexAttr(8), rewriter.getIndexAttr(2)};
                      auto subview = rewriter.create<memref::SubViewOp>(
                          kForOp.getLoc(), rhsClone->getResult(0), offsets,
                          sizes, strides);
                      oddB = rewriter.create<
                          mlir::x86vector::CvtPackedOddIndexedToF32Op>(
                          kForOp.getLoc(), dstType, subview);
                    }

                    // Odd FMAs
                    for (int i = 0; i < M; i++) {
                      auto fmaOdd = rewriter.create<vector::FMAOp>(
                          kForOp.getLoc(), matf32[i], oddB,
                          iterArgsNewKForOp[k]);
                      k++;
                      oddFMAs.push_back(fmaOdd);
                    }
                  }

                  // Clear A matrix  odd loads to save even loads
                  matf32.clear();

                  // Load even elements of A Matrix and store in a DS
                  for (int i = 0; i < M; i++) {
                    Value evenA;

                    if (adl) {
                      Value indexOp_ci =
                          rewriter.create<arith::ConstantIndexOp>(
                              reductionForOp.getLoc(), i);
                      auto valueRow = rewriter.create<vector::LoadOp>(
                          kForOp.getLoc(), VectorType::get(2, elementType),
                          lhsClone->getResult(0),
                          ValueRange{indexOp_c0, indexOp_ci, indexOp_c0,
                                     indexOp_c0});
                      auto bitcast_i32 = rewriter.create<vector::BitCastOp>(
                          kForOp.getLoc(), VectorType::get(1, i32Type),
                          valueRow);
                      auto shift = rewriter.create<arith::ShLIOp>(
                          kForOp.getLoc(), VectorType::get(1, i32Type),
                          bitcast_i32,
                          rewriter.create<vector::BroadcastOp>(
                              kForOp.getLoc(), VectorType::get(1, i32Type),
                              cst16));
                      auto bcstShift =
                          rewriterNewKForOp.create<vector::BroadcastOp>(
                              kForOp.getLoc(),
                              VectorType::get(sizeFactor, i32Type), shift);
                      evenA = rewriter.create<vector::BitCastOp>(
                          kForOp.getLoc(),
                          VectorType::get(sizeFactor, rewriter.getF32Type()),
                          bcstShift);
                    }

                    if (srf) {
                      SmallVector<OpFoldResult> offsets = {
                          rewriter.getIndexAttr(0),
                          rewriter.getIndexAttr(i),
                          rewriter.getIndexAttr(0),
                          rewriter.getIndexAttr(0),
                      };

                      auto subview = rewriter.create<memref::SubViewOp>(
                          kForOp.getLoc(), lhsClone->getResult(0), offsets,
                          sizes, strides);
                      evenA =
                          rewriter.create<mlir::x86vector::BcstToPackedF32Op>(
                              kForOp.getLoc(), dstType, subview);
                    }

                    matf32.push_back(evenA);
                  }

                  // Load even elements of B-Matrix, perform fma (even), and
                  // store to a DS
                  if (adl) {
                    for (int j = 0, k = 0; j < N; j = j + 16) {

                      Value indexOp_cj =
                          rewriter.create<arith::ConstantIndexOp>(
                              reductionForOp.getLoc(), j);
                      auto valueRow = rewriter.create<vector::LoadOp>(
                          kForOp.getLoc(), VectorType::get(32, elementType),
                          rhsClone->getResult(0),
                          ValueRange{indexOp_c0, indexOp_c0, indexOp_cj,
                                     indexOp_c0});
                      auto bitcast_i32 = rewriter.create<vector::BitCastOp>(
                          kForOp.getLoc(), VectorType::get(16, i32Type),
                          valueRow);
                      auto shift = rewriter.create<arith::ShLIOp>(
                          kForOp.getLoc(), VectorType::get(16, i32Type),
                          bitcast_i32,
                          rewriter.create<vector::BroadcastOp>(
                              kForOp.getLoc(), VectorType::get(16, i32Type),
                              cst16));
                      auto evenB_16 = rewriter.create<vector::BitCastOp>(
                          kForOp.getLoc(),
                          VectorType::get(16, rewriter.getF32Type()), shift);

                      auto offsetsAttr = rewriter.getI64ArrayAttr({0});
                      auto offsetsAttr_8 =
                          rewriter.getI64ArrayAttr({sizeFactor});
                      auto sizesAttr = rewriter.getI64ArrayAttr({sizeFactor});
                      auto stridesAttr = rewriter.getI64ArrayAttr({1});
                      auto evenB1 =
                          rewriter.create<vector::ExtractStridedSliceOp>(
                              kForOp.getLoc(),
                              VectorType::get(sizeFactor,
                                              rewriter.getF32Type()),
                              evenB_16, offsetsAttr, sizesAttr, stridesAttr);
                      auto evenB2 =
                          rewriter.create<vector::ExtractStridedSliceOp>(
                              kForOp.getLoc(),
                              VectorType::get(sizeFactor,
                                              rewriter.getF32Type()),
                              evenB_16, offsetsAttr_8, sizesAttr, stridesAttr);

                      // Even FMAs
                      for (int i = 0; i < M; i++) {
                        auto fmaEven = rewriter.create<vector::FMAOp>(
                            kForOp.getLoc(), matf32[i], evenB1, oddFMAs[k]);
                        k++;
                        evenFMAs.push_back(fmaEven);
                      }

                      for (int i = 0; i < M; i++) {
                        auto fmaEven = rewriter.create<vector::FMAOp>(
                            kForOp.getLoc(), matf32[i], evenB2, oddFMAs[k]);
                        k++;
                        evenFMAs.push_back(fmaEven);
                      }
                    }
                  }

                  if (srf) {
                    for (int j = 0, k = 0; j < N; j = j + sizeFactor) {

                      SmallVector<OpFoldResult> offsets = {
                          rewriter.getIndexAttr(0),
                          rewriter.getIndexAttr(0),
                          rewriter.getIndexAttr(j),
                          rewriter.getIndexAttr(0),
                      };
                      llvm::SmallVector<OpFoldResult> sizes = {
                          rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
                          rewriter.getIndexAttr(8), rewriter.getIndexAttr(2)};
                      auto subview = rewriter.create<memref::SubViewOp>(
                          kForOp.getLoc(), rhsClone->getResult(0), offsets,
                          sizes, strides);
                      auto evenB = rewriter.create<
                          mlir::x86vector::CvtPackedEvenIndexedToF32Op>(
                          kForOp.getLoc(), dstType, subview);

                      // Even FMAs
                      for (int i = 0; i < M; i++) {
                        auto fmaEven = rewriter.create<vector::FMAOp>(
                            kForOp.getLoc(), matf32[i], evenB, oddFMAs[k]);
                        k++;
                        evenFMAs.push_back(fmaEven);
                      }
                    }
                  }
                }

                rewriterNewKForOp.create<scf::YieldOp>(locNewKForOp, evenFMAs);
              });

          rewriterNewReductionForOp.create<scf::YieldOp>(
              locNewReductionForOp, newKForOp.getResults());
        });

    // Check the mlp pattern
    arith::AddFOp addOp;
    arith::MaximumFOp maxOp;
    memref::SubViewOp subview_readOp;
    memref::GetGlobalOp global_readOp;

    auto zeroAttr = rewriter.getFloatAttr(rewriter.getF32Type(), 0.0);
    auto denseAttr = DenseElementsAttr::get(
        VectorType::get(sizeFactor, rewriter.getF32Type()), zeroAttr);
    auto cst_zero = rewriter.create<arith::ConstantOp>(
        reductionForOp.getLoc(),
        VectorType::get(sizeFactor, rewriter.getF32Type()), denseAttr);
    auto c1 = rewriter.create<arith::ConstantOp>(
        kForOp.getLoc(),
        DenseIntElementsAttr::get(VectorType::get(sizeFactor, i32Type), 1));
    auto c16 = rewriter.create<arith::ConstantOp>(
        kForOp.getLoc(),
        DenseIntElementsAttr::get(VectorType::get(sizeFactor, i32Type), 16));
    auto c7fff = rewriter.create<arith::ConstantOp>(
        kForOp.getLoc(), DenseIntElementsAttr::get(
                             VectorType::get(sizeFactor, i32Type), 0x7fff));

    auto subview_itr = subviewOpAcc.getSource();
    Value subview_tmp = subviewOpAcc;

    while (true) {
      if (auto subview_2 = subview_itr.getDefiningOp<memref::SubViewOp>()) {

        if (subview_2.getType().getRank() != 2) {
          break;
        }
        subview_tmp = subview_itr;
        subview_itr = subview_2.getSource();
        continue;
      }
      break;
    }

    Value source;
    if (subview_itr.getType().getRank() == 2) {
      source = subview_itr;
    } else {
      source = subview_tmp;
    }

    // get the addOp and maxOp source
    for (auto users : source.getUsers()) {
      if (auto vectorRead = dyn_cast<vector::TransferReadOp>(users)) {
        for (auto read_users : vectorRead->getUsers()) {
          if (auto add_Op = dyn_cast<arith::AddFOp>(read_users)) {
            addOp = add_Op;
          }
          if (auto max_Op = dyn_cast<arith::MaximumFOp>(read_users)) {
            if (addOp)
              maxOp = max_Op;
          }
        }
      }
    }

    // get the input source for addOp via vector transfer read
    if (addOp && maxOp) {
      vector::TransferReadOp readOp_add;
      if (auto vectBcst = addOp.getLhs().getDefiningOp<vector::BroadcastOp>()) {
        if (auto vectorRead =
                vectBcst.getSource().getDefiningOp<vector::TransferReadOp>()) {
          readOp_add = vectorRead;
        }
      }

      if (readOp_add) {
        if (auto subview =
                readOp_add.getBase().getDefiningOp<memref::SubViewOp>()) {
          subview_readOp = subview;
        }

        if (auto global_read =
                readOp_add.getBase().getDefiningOp<memref::GetGlobalOp>()) {
          global_readOp = global_read;
        }
      }
    }

    // B-matrix (N) induction variable
    auto nInductionVar = nForOp.getInductionVar();

    for (int j = 0, k = 0; j < N; j = j + sizeFactor) {
      for (int i = 0; i < M; i++) {
        Value indexOp =
            rewriter.create<arith::ConstantIndexOp>(reductionForOp.getLoc(), i);
        Value indexOp_B =
            rewriter.create<arith::ConstantIndexOp>(reductionForOp.getLoc(), j);
        Type type;
        if (elementType.isBF16())
          type = rewriter.getBF16Type();
        if (elementType.isF16())
          type = rewriter.getF16Type();

        auto acc_value = newReductionForOp.getResult(k);
        k++;

        if (addOp && maxOp) {
          Value add_row;

          if (global_readOp) {
            auto index_mlp = rewriter.create<arith::AddIOp>(
                reductionForOp.getLoc(), rewriter.getIndexType(), nInductionVar,
                indexOp_B);
            add_row = rewriter.create<vector::LoadOp>(
                reductionForOp.getLoc(),
                VectorType::get(sizeFactor, elementType), global_readOp,
                ValueRange{index_mlp});
          }

          if (subview_readOp) {
            // auto nInductionVar = nForOp.getInductionVar();
            auto index_mlp = rewriter.create<arith::AddIOp>(
                reductionForOp.getLoc(), rewriter.getIndexType(), nInductionVar,
                indexOp_B);
            auto offsetsVec = subview_readOp.getMixedOffsets();
            llvm::ArrayRef<mlir::OpFoldResult> offsets = offsetsVec;
            auto val_offset = offsets[0].dyn_cast<mlir::Value>();
            add_row = rewriter.create<vector::LoadOp>(
                reductionForOp.getLoc(),
                VectorType::get(sizeFactor, elementType),
                subview_readOp.getSource(), ValueRange{val_offset, index_mlp});
          }

          // Fused mlp happens here
          if (add_row) {
            Value f32MLPVector;
            if (elementType.isBF16()) {
              auto bitcast_i16 = rewriter.create<vector::BitCastOp>(
                  reductionForOp.getLoc(), VectorType::get(sizeFactor, i16Type),
                  add_row);
              auto extend_i32 = rewriter.create<arith::ExtUIOp>(
                  reductionForOp.getLoc(), VectorType::get(sizeFactor, i32Type),
                  bitcast_i16);
              auto shiftOp = rewriter.create<arith::ShLIOp>(
                  reductionForOp.getLoc(), VectorType::get(sizeFactor, i32Type),
                  extend_i32,
                  rewriter.create<vector::BroadcastOp>(
                      reductionForOp.getLoc(),
                      VectorType::get(sizeFactor, i32Type), cst16));
              f32MLPVector = rewriter.create<vector::BitCastOp>(
                  reductionForOp.getLoc(),
                  VectorType::get(sizeFactor, rewriter.getF32Type()), shiftOp);
            }

            if (elementType.isF16()) {
              f32MLPVector = rewriter.create<arith::ExtFOp>(
                  reductionForOp.getLoc(),
                  VectorType::get(sizeFactor, rewriter.getF32Type()), add_row);
            }

            auto add = rewriter.create<arith::AddFOp>(
                reductionForOp.getLoc(),
                mlir::VectorType::get(sizeFactor, rewriter.getF32Type()),
                acc_value, f32MLPVector);
            auto max = rewriter.create<arith::MaximumFOp>(
                reductionForOp.getLoc(),
                mlir::VectorType::get(sizeFactor, rewriter.getF32Type()), add,
                cst_zero);
            acc_value = max;
          }
        }

        Value vec_final;

        // We do f32 -> bf16 downconvert using rshift, truncate and rounding the
        // lsb for ADL
        if (adl || clx) {
          auto vec = rewriter.create<vector::BitCastOp>(
              kForOp.getLoc(), VectorType::get(sizeFactor, i32Type), acc_value);
          auto rshift = rewriter.create<arith::ShRUIOp>(
              kForOp.getLoc(), VectorType::get(sizeFactor, i32Type), vec, c16);
          auto leastSB = rewriter.create<arith::AndIOp>(reductionForOp.getLoc(),
                                                        rshift, c1);
          auto roundBias = rewriter.create<arith::AddIOp>(
              reductionForOp.getLoc(), c7fff, leastSB);
          auto rounded_vec = rewriter.create<arith::AddIOp>(
              reductionForOp.getLoc(), vec, roundBias);
          auto shift = rewriter.create<arith::ShRUIOp>(reductionForOp.getLoc(),
                                                       rounded_vec, c16);
          auto truncate = rewriter.create<arith::TruncIOp>(
              reductionForOp.getLoc(), VectorType::get(sizeFactor, i16Type),
              shift);
          vec_final = rewriter.create<vector::BitCastOp>(
              reductionForOp.getLoc(),
              VectorType::get(sizeFactor, rewriter.getBF16Type()), truncate);
        }

        // We do arith.tuncf for f32 -> bf16 in SRF/ARL
        if (srf || cpx) {
          vec_final = rewriter.create<arith::TruncFOp>(
              reductionForOp.getLoc(), VectorType::get(sizeFactor, type),
              acc_value);
        }

        rewriter.create<vector::StoreOp>(reductionForOp.getLoc(), vec_final,
                                         subviewOpAcc,
                                         ValueRange{indexOp, indexOp_B});
      }
    }

    // Remove the transfer write operations
    if (addOp && maxOp && (subview_readOp || global_readOp)) {
      Operation *writeOp_add;
      for (auto val : addOp->getUsers()) {
        writeOp_add = dyn_cast<vector::TransferWriteOp>(val);
        if (writeOp_add) {
          rewriter.eraseOp(writeOp_add);
          break;
        }
      }

      Operation *writeOp_max;
      for (auto val : maxOp->getUsers()) {
        writeOp_max = dyn_cast<vector::TransferWriteOp>(val);
        if (writeOp_max) {
          rewriter.eraseOp(writeOp_max);
          break;
        }
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

void populateBF16KERNELSPatterns(RewritePatternSet &patterns) {
  patterns.add<BF16KERNELSOp>(patterns.getContext());
}

struct BF16KERNELS : public impl::BF16KERNELSBase<BF16KERNELS> {
  using BF16KERNELSBase::BF16KERNELSBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateBF16KERNELSPatterns(patterns);
    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingOps);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
  }
};
} // namespace tpp
} // namespace mlir

