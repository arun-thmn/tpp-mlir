//===- ConvertVectorToXsmm.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Xsmm/XsmmOps.h"
#include "TPP/Dialect/Xsmm/XsmmUtils.h"
#include "TPP/IR/MatcherUtils.h"
#include "TPP/IR/StructuredOpMatcher.h"
#include "TPP/Passes.h"
#include "TPP/Transforms/Transforms.h"
#include "TPP/Transforms/Utils/TransformUtils.h"
#include "TPP/Transforms/Utils/VNNIUtils.h"
#include "TPP/Transforms/Utils/ValueUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace vector;
using namespace structured_match;
using namespace linalg;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_CONVERTVECTORTOXSMM
#define GEN_PASS_DECL_CONVERTVECTORTOXSMM
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

#define DEBUG_TYPE "convert-vector-to-xsmm"

namespace mlir {
namespace tpp {

struct ConvertVectorToXsmm
    : public impl::ConvertVectorToXsmmBase<ConvertVectorToXsmm> {
  void runOnOperation() override;
};

template <typename OpTy>
std::function<bool(Operation *op)> FuncType =
    [](Operation *op) { return isa<OpTy>(op); };

static bool
WithInputs(vector::ContractionOp op,
           SmallVector<std::function<bool(Operation *op)>> operations,
           SmallVector<OpOperand *> &inputs) {

  auto scfForallOp = cast<scf::ForallOp>(op->getParentOp());
  if (!scfForallOp)
    return false;
  auto region = &scfForallOp->getRegion(0);
  if (!region->hasOneBlock())
    return false;

  for (int i = 0; i < operations.size(); i++) {
    auto input = op.getOperand(i);
    if (!operations[i](input.getDefiningOp()))
      return false;
    if (input.getDefiningOp()->getParentOp() != scfForallOp)
      return false;
    assert(isa<memref::SubViewOp>(
               input.getDefiningOp()->getOperand(0).getDefiningOp()) ||
           isa<memref::GetGlobalOp>(
               input.getDefiningOp()->getOperand(0).getDefiningOp()));
    inputs.push_back(&input.getDefiningOp()->getOpOperand(0));
  }
  return true;
}

static bool WithOutput(vector::ContractionOp op,
                       std::function<bool(Operation *op)> operation,
                       SmallVector<OpOperand *> &output) {
  auto scfForallOp = cast<scf::ForallOp>(op->getParentOp());
  if (!scfForallOp)
    return false;
  auto region = &scfForallOp->getRegion(0);
  if (!region->hasOneBlock())
    return false;

  // Check on the inner chain of operations in the right order.
  // Make sure all operands are used and chained
  for (auto use : op.getResult().getUsers()) {
    if (use != op && operation(use) && use->getParentOp() == scfForallOp) {
      assert(isa<memref::SubViewOp>(use->getOperand(1).getDefiningOp()));
      output.push_back(&use->getOpOperand(1));
      return true;
    }
  }
  return false;
}

static bool
WithOps(Region *region, Operation *op,
        SmallVector<std::function<bool(Operation *op)>> operations) {
  // Basic checks
  if (!isa<scf::ForallOp>(op))
    return false;
  auto scfForallOp = cast<scf::ForallOp>(op);
  if (!region->hasOneBlock())
    return false;
  auto &block = region->front();

  llvm::SmallSetVector<Value, 4> chainedValues;

  auto start = block.begin();
  for (auto opItr = block.begin(); opItr != block.end(); opItr++) {
    if (!operations[0](&*opItr))
      continue;
    start = opItr;
    break;
  }
  // Check on the inner chain of operations in the right order.
  // Make sure all operands are used and chained
  for (auto check : operations) {
    Operation *innerOp = &*start;
    // Must be right op in right order
    if (start == block.end() || !check(innerOp))
      return false;
    start++;
    // At least one operand must come from args or a previous op
    bool consumesValueFromChain = false;
    if (chainedValues.empty()) {
      consumesValueFromChain = true;
    } else {
      for (auto operand : innerOp->getOperands()) {
        if (chainedValues.contains(operand)) {
          chainedValues.remove(operand);
          consumesValueFromChain = true;
        }
      }
    }

    // Operation isn't in the chain
    if (!consumesValueFromChain)
      return false;

    for (auto ret : innerOp->getResults()) {
      chainedValues.insert(ret);
    }
  }

  return true;
}

struct BrgemmInfo {
  int64_t m;
  int64_t n;
  int64_t k;
  int64_t batch;

  int64_t lda;
  int64_t ldb;
  int64_t ldc;
  int64_t strideA;
  int64_t strideB;

  bool isVnni = false;
};

// Return the position of `dim` in the codomain of `operand`.
std::optional<unsigned> getPosInCodomain(unsigned dim, OpOperand *operand,
                                         vector::ContractionOp contractOp,
                                         AffineMap map) {
  return map.getResultPosition(getAffineDimExpr(dim, contractOp.getContext()));
}

static void replaceOpWithGemmLikeOp(RewriterBase &rewriter,
                                    vector::ContractionOp contractOp,
                                    BrgemmInfo brgemmInfo,
                                    SmallVector<OpOperand *> inputs) {
  OpBuilder::InsertionGuard guard(rewriter);
  auto m = brgemmInfo.m;
  auto n = brgemmInfo.n;
  auto k = brgemmInfo.k;
  auto batch = brgemmInfo.batch;
  int64_t lda = brgemmInfo.lda;
  int64_t ldb = brgemmInfo.ldb;
  int64_t ldc = brgemmInfo.ldc;
  int64_t strideA = brgemmInfo.strideA;
  int64_t strideB = brgemmInfo.strideB;

  auto dtype =
      xsmm::utils::getDataType(rewriter, contractOp.getOperand(0).getType());
  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  Location loc = contractOp.getLoc();
  xsmm::GemmFlagsAttr gemmFlags;
  if (brgemmInfo.isVnni) {
    gemmFlags = xsmm::GemmFlagsAttr::get(rewriter.getContext(),
                                         xsmm::GemmFlags::VNNI_B);
  } else {
    gemmFlags =
        xsmm::GemmFlagsAttr::get(rewriter.getContext(), xsmm::GemmFlags::NONE);
  }
  auto flags = rewriter.getArrayAttr(gemmFlags);
  SmallVector<Value> invokeOperands;

  if (batch != 0) {
    DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
        rewriter.getContext(),
        ArrayRef<int64_t>{m, n, k, lda, ldb, ldc, strideA, strideB});
    Value dispatched = rewriter.create<xsmm::BrgemmDispatchOp>(
        loc, integer64, dims, flags, dtype);
    Value batchDim = rewriter.create<arith::ConstantOp>(
        loc, integer64, rewriter.getIntegerAttr(integer64, batch));
    invokeOperands.push_back(dispatched);
    for (auto operand : inputs) {
      invokeOperands.push_back(operand->get());
    }
    invokeOperands.push_back(batchDim);
    rewriter.create<xsmm::BrgemmOp>(loc, dtype, invokeOperands);
    for (auto user : contractOp->getResult(0).getUsers()) {
      rewriter.eraseOp(user);
    }
    rewriter.eraseOp(contractOp);

  } else {
    DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
        rewriter.getContext(), ArrayRef<int64_t>{m, n, k, lda, ldb, ldc});
    Value dispatched = rewriter.create<xsmm::GemmDispatchOp>(
        loc, integer64, dims, flags, dtype);
    invokeOperands.push_back(dispatched);
    for (auto operand : inputs) {
      invokeOperands.push_back(operand->get());
    }
    rewriter.replaceOpWithNewOp<xsmm::GemmOp>(contractOp, dtype,
                                              invokeOperands);
    rewriter.create<xsmm::GemmOp>(loc, dtype, invokeOperands);
    for (auto user : contractOp->getResult(0).getUsers()) {
      rewriter.eraseOp(user);
    }

    rewriter.eraseOp(contractOp);
  }
}

// Structural matcher.
static FailureOr<ContractionDimensions> checkStructure(
    vector::ContractionOp contractOp, SmallVector<OpOperand *> &inputs,
    SmallVector<OpOperand *> &outputs, ArrayRef<AffineMap> indexingMap) {
  if (!HasStaticShape()(inputs[0], inputs[0]->get().getDefiningOp()) ||
      !HasStaticShape()(inputs[1], inputs[1]->get().getDefiningOp()) ||
      !HasStaticShape()(inputs[2], inputs[2]->get().getDefiningOp()) ||
      !HasStaticShape()(outputs[0], outputs[0]->get().getDefiningOp()) ||
      !HasStaticStrides()(inputs[0], inputs[0]->get().getDefiningOp()) ||
      !HasStaticStrides()(inputs[1], inputs[1]->get().getDefiningOp()) ||
      !HasStaticStrides()(inputs[2], inputs[2]->get().getDefiningOp()) ||
      !HasStaticStrides()(outputs[0], outputs[0]->get().getDefiningOp())) {
    return failure();
  }

  return inferContractionDims(indexingMap);
}

static SmallVector<int64_t, 4>
createFlatListOfOperandStaticDims(vector::ContractionOp contractOp) {
  SmallVector<int64_t, 4> res;
  for (OpOperand &opOperand : contractOp.getOperation()->getOpOperands())
    llvm::append_range(
        res, dyn_cast<VectorType>(opOperand.get().getType()).getShape());
  return res;
}

static SmallVector<int64_t, 4>
computeStaticLoopSizes(vector::ContractionOp contractOp,
                       ArrayRef<AffineMap> maps) {
  AffineMap map = concatAffineMaps(maps);
  unsigned numDims = map.getNumDims(), numRes = map.getNumResults();
  SmallVector<int64_t, 4> allShapeSizes =
      createFlatListOfOperandStaticDims(contractOp);
  SmallVector<int64_t, 4> res(numDims, 0);
  for (unsigned idx = 0; idx < numRes; ++idx) {
    auto result = map.getResult(idx);
    if (auto d = dyn_cast<AffineDimExpr>(result))
      res[d.getPosition()] = allShapeSizes[idx];
  }
  return res;
}

// Access matcher.
static FailureOr<BrgemmInfo> checkAccess(vector::ContractionOp contractOp,
                                         unsigned m, unsigned n, unsigned k,
                                         std::optional<unsigned> batchPos,
                                         SmallVector<OpOperand *> inputs,
                                         ArrayRef<AffineMap> indexingMap) {
  OpOperand *operandA = inputs[0];
  OpOperand *operandB = inputs[1];
  OpOperand *operandC = inputs[2];

  auto checkStridesAndGetLda = [&](unsigned minorDim, unsigned majorDim,
                                   OpOperand *operand,
                                   AffineMap map) -> FailureOr<int64_t> {
    auto minorDimPosInCodomain =
        getPosInCodomain(minorDim, operand, contractOp, map);
    auto majorDimPosInCodomain =
        getPosInCodomain(majorDim, operand, contractOp, map);
    if (!minorDimPosInCodomain || !majorDimPosInCodomain)
      return failure();
    auto stridesOnOperand = utils::getStaticStrides(operand->get());
    if (failed(stridesOnOperand) ||
        (*stridesOnOperand)[*minorDimPosInCodomain] != 1) {
      return failure();
    }
    return (*stridesOnOperand)[*majorDimPosInCodomain];
  };

  // A(m, k)
  auto lda = checkStridesAndGetLda(k, m, operandA, indexingMap[0]);
  if (failed(lda)) {
    return failure();
  }
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] Strides on A: OK\n");

  // B(k, n)
  auto ldb = checkStridesAndGetLda(n, k, operandB, indexingMap[1]);
  if (failed(ldb)) {
    return failure();
  }
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] Strides on B: OK\n");

  // C(m, n)
  auto ldc = checkStridesAndGetLda(n, m, operandC, indexingMap[2]);
  if (failed(ldc)) {
    return failure();
  }
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] Strides on C: OK\n");

  int64_t strideA = 1;
  int64_t strideB = 1;
  if (batchPos) {
    auto batchPosCodomainA = getPosInCodomain(batchPos.value(), operandA,
                                              contractOp, indexingMap[0]);
    auto stridesOnA = utils::getStaticStrides(operandA->get());
    strideA = (*stridesOnA)[*batchPosCodomainA];

    auto batchPosCodomainB = getPosInCodomain(batchPos.value(), operandB,
                                              contractOp, indexingMap[1]);
    auto stridesOnB = utils::getStaticStrides(operandB->get());
    strideB = (*stridesOnB)[*batchPosCodomainB];
  }

  auto loops = computeStaticLoopSizes(contractOp, indexingMap);
  int64_t batchVal = (batchPos) ? loops[batchPos.value()] : 0;

  BrgemmInfo info{loops[m], loops[n], loops[k], batchVal, *lda,
                  *ldb,     *ldc,     strideA,  strideB};
  return info;
}

// Check if the given generic is mappable to a brgemm xsmm op.
// - It is a contraction, with:
// -- 1 m and 1 n and 2 k dimensions.
// -- m appears on the LHS and OUT but not in RHS.
// -- n appears on the RHS and OUT but not in LHS.
// -- k and k' appear on the RHS and LHS but not OUT.
// -- the stride of the minor dimension for A, k is 1.
// -- the stride of the minor dimension for B, n is 1.
// -- the stride of the minor dimension for C, n is 1.
static FailureOr<BrgemmInfo> isMappableToBrgemm(
    vector::ContractionOp contractOp, SmallVector<OpOperand *> &inputs,
    SmallVector<OpOperand *> &output, ArrayRef<AffineMap> indexingMap) {
  auto contractionDims =
      checkStructure(contractOp, inputs, output, indexingMap);
  if (failed(contractionDims)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[isMappableToBrgemm] Failed on checkStructure\n");
    return failure();
  }
  unsigned m = contractionDims->m[0];
  unsigned n = contractionDims->n[0];
  unsigned k = contractionDims->k.back();
  std::optional<unsigned> batch;
  if (contractionDims->k.size() == 2)
    batch = contractionDims->k.front();

  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] Candidate dims: "
                          << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] m: " << m << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] n: " << n << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] k: " << k << "\n");
  if (batch)
    LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] batch: " << batch << "\n");
  else
    LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] no batch dim\n");

  return checkAccess(contractOp, m, n, k, batch, inputs, indexingMap);
}

// Convert vector.contract to a XSMM brgemm op.
struct ConvertVectorContractToBatchReduceMatmul
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {

    SmallVector<std::function<bool(Operation * op)>> operations;
    operations.push_back(FuncType<vector::ContractionOp>);
    if (!WithOps(&contractOp->getParentOp()->getRegion(0),
                 contractOp->getParentOp(), operations)) {
      return failure();
    }

    SmallVector<std::function<bool(Operation * op)>> inputOperations;
    inputOperations.push_back(FuncType<vector::TransferReadOp>);
    inputOperations.push_back(FuncType<vector::TransferReadOp>);
    inputOperations.push_back(FuncType<vector::TransferReadOp>);
    SmallVector<OpOperand *> inputs;

    if (!WithInputs(contractOp, inputOperations, inputs)) {
      return failure();
    }

    SmallVector<OpOperand *> outputs;
    if (!WithOutput(contractOp, FuncType<vector::TransferWriteOp>, outputs)) {
      return failure();
    }
    if (contractOp.getKind() != CombiningKind::ADD) {
      return failure();
    }
    auto iteratorTypes = contractOp.getIteratorTypes();
    if (iteratorTypes.size() != 4)
      return failure();
    size_t size = iteratorTypes.size() - 1;
    bool match = vector::isReductionIterator(iteratorTypes[size]) &&
                 vector::isParallelIterator(iteratorTypes[size - 1]) &&
                 vector::isParallelIterator(iteratorTypes[size - 2]) &&
                 vector::isReductionIterator(iteratorTypes[size - 3]);
    if (!match)
      return failure();

    auto indexingMaps = contractOp.getIndexingMaps();
    if (indexingMaps.size() != 3)
      return failure();
    AffineExpr r1, p3, p4, r2;
    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    auto infer = [&](MapList m, PatternRewriter &rewriter) {
      return rewriter.getAffineMapArrayAttr(
          AffineMap::inferFromExprList(m, contractOp.getContext()));
    };

    bindDims(contractOp.getContext(), r1, p3, p4, r2);
    auto expectedMaps = infer({{r1, p3, r2}, {r1, r2, p4}, {p3, p4}}, rewriter);

    if (indexingMaps != expectedMaps)
      return failure();
    ArrayRef<SmallVector<AffineExpr, 4>> map = {
        {r1, p3, r2}, {r1, r2, p4}, {p3, p4}};
    ArrayRef<AffineMap> affineMaps =
        AffineMap::inferFromExprList(map, contractOp.getContext());
    auto brgemmInfo =
        isMappableToBrgemm(contractOp, inputs, outputs, affineMaps);
    if (failed(brgemmInfo)) {
      return failure();
    }
    replaceOpWithGemmLikeOp(rewriter, contractOp, *brgemmInfo, inputs);
    return success();
  }
};

void populateVectorToXsmmPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertVectorContractToBatchReduceMatmul>(patterns.getContext());
}

void ConvertVectorToXsmm::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateVectorToXsmmPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

} // namespace tpp
} // namespace mlir
