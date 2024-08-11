//===- ConvertVectorToXsmm.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Xsmm/XsmmOps.cpp.inc"
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
#include <iostream>

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
WithInputs(PatternRewriter &rewriter, Operation *op,
           SmallVector<std::function<bool(Operation *op)>> operations,
           SmallVector<OpOperand *> &inputs) {

  auto scfForallOp = cast<scf::ForallOp>(op->getParentOp());
  if (!scfForallOp)
    return false;
  auto region = &scfForallOp->getRegion(0);
  if (!region->hasOneBlock())
    return false;

  for (int i = 0; i < operations.size(); i++) {
    auto input = op->getOperand(i);
    if (!operations[i](input.getDefiningOp()))
      return false;
    if (input.getDefiningOp()->getParentOp() != scfForallOp)
      return false;
    auto dataType = xsmm::utils::getDataType(rewriter, input.getType());
    if (dataType ==
        xsmm::DataTypeAttr::get(rewriter.getContext(), xsmm::DataType::BF16)) {
      assert(isa<memref::ExpandShapeOp>(
                 input.getDefiningOp()->getOperand(0).getDefiningOp()) ||
             isa<memref::GetGlobalOp>(
                 input.getDefiningOp()->getOperand(0).getDefiningOp()) ||
             isa<memref::SubViewOp>(
                 input.getDefiningOp()->getOperand(0).getDefiningOp()) ||
             isa<vector::TransferReadOp>(
                 input.getDefiningOp()->getOperand(0).getDefiningOp()));

    } else {
      assert(isa<memref::SubViewOp>(
                 input.getDefiningOp()->getOperand(0).getDefiningOp()) ||
             isa<memref::GetGlobalOp>(
                 input.getDefiningOp()->getOperand(0).getDefiningOp()) ||
             isa<vector::TransferReadOp>(
                 input.getDefiningOp()->getOperand(0).getDefiningOp()));
    }
    inputs.push_back(&input.getDefiningOp()->getOpOperand(0));
  }
  return true;
}

static bool WithOutput(Operation *op,
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
  for (auto use : op->getResult(0).getUsers()) {
    if (use != op && operation(use) && use->getParentOp() == scfForallOp) {
      assert(isa<memref::SubViewOp>(use->getOperand(1).getDefiningOp()));
      output.push_back(&use->getOpOperand(1));
      return true;
    }
  }
  return false;
}

static bool WithOps(Region *region, Operation *op,
                    SmallVector<std::function<bool(Operation *op)>> operations,
                    SmallVector<Operation *> &opChain) {
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
    opChain.push_back(&*opItr);
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

static Operation *WithZeroInit(OpOperand *input,
                               vector::TransferWriteOp &transferWriteOp) {
  Operation *rootOp = nullptr;
  for (auto user : input->get().getUsers()) {
    if (isa<vector::TransferWriteOp>(user) &&
        utils::isValConstZero(
            dyn_cast<vector::TransferWriteOp>(user).getOperand(0))) {
      transferWriteOp = dyn_cast<vector::TransferWriteOp>(user);
      rootOp = transferWriteOp;
      break;
    }
  }
  assert(rootOp != nullptr);
  Value dest = rootOp->getOperands().back();
  DenseSet<Operation *> destUsers(dest.getUsers().begin(),
                                  dest.getUsers().end());

  Block *blck = nullptr;
  if (auto bbArg = dyn_cast<BlockArgument>(dest)) {
    blck = bbArg.getOwner();
  } else {
    Operation *defOp = dest.getDefiningOp();
    if (!defOp)
      return nullptr;
    ;
    blck = defOp->getBlock();
  }
  assert(blck && "must be a valid ptr");
  auto it = blck->begin();
  auto itEnd = blck->end();
  while (it != itEnd && &*it != rootOp) {
    // View may introduce aliasing.
    if (auto view = dyn_cast<ViewLikeOpInterface>(&*it)) {
      if (view.getViewSource() == dest)
        return nullptr;
    }
    it++;
  }

  if (it == itEnd)
    return nullptr;

  while (++it != itEnd) {
    // Skip operations that do not touch `dest`.
    if (!destUsers.count(&*it))
      continue;
    // No memory effects other than read.
    if (mlir::hasSingleEffect<MemoryEffects::Read>(&*it, dest))
      continue;
    // View may introduce aliasing.
    if (auto view = dyn_cast<ViewLikeOpInterface>(&*it)) {
      if (view.getViewSource() == dest)
        return nullptr;
    }
    // A gemm or brgemm operation touching `dest`, fold if the
    // output (i.e. C matrix) is `dest`.
    if (auto gemmOp = dyn_cast<xsmm::GemmOp>(*it)) {
      Value outVal = gemmOp.getOutput();
      if (outVal == dest)
        break;
    }
    if (auto brgemmOp = dyn_cast<xsmm::BrgemmOp>(*it)) {
      Value outVal = brgemmOp.getOutput();
      if (outVal == dest)
        break;
    }
    if (auto fusedBrgemmOp = dyn_cast<xsmm::FusedBrgemmOp>(*it)) {
      Value outVal = fusedBrgemmOp.getOutput();
      if (outVal == dest)
        break;
    }
    // Fail.
    return nullptr;
  }
  if (it == itEnd)
    return nullptr;
  return &*it;
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

static std::optional<Operation *>
replaceOpWithGemmLikeOp(RewriterBase &rewriter,
                        vector::ContractionOp contractOp, BrgemmInfo brgemmInfo,
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
    auto brgemmOp = rewriter.create<xsmm::BrgemmOp>(loc, dtype, invokeOperands);
    for (auto user : contractOp->getResult(0).getUsers()) {
      rewriter.eraseOp(user);
    }
    rewriter.eraseOp(contractOp);
    return brgemmOp;
  } else {
    DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
        rewriter.getContext(), ArrayRef<int64_t>{m, n, k, lda, ldb, ldc});
    Value dispatched = rewriter.create<xsmm::GemmDispatchOp>(
        loc, integer64, dims, flags, dtype);
    invokeOperands.push_back(dispatched);
    for (auto operand : inputs) {
      invokeOperands.push_back(operand->get());
    }
    auto gemmOp = rewriter.create<xsmm::GemmOp>(loc, dtype, invokeOperands);
    for (auto user : contractOp->getResult(0).getUsers()) {
      rewriter.eraseOp(user);
    }
    rewriter.eraseOp(contractOp);
    return gemmOp;
  }
}

static std::optional<Operation *> replaceOpWithFusedBrgemmOp(
    RewriterBase &rewriter, vector::ContractionOp contractOp,
    BrgemmInfo brgemmInfo, SmallVector<OpOperand *> inputs,
    xsmm::UnaryKind unaryKind, xsmm::BinaryKind binaryKind, OpOperand *bias,
    Operation *unaryOp, Operation *binaryOp, OpOperand *binaryOpResult,
    Operation *zeroOp) {
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
  SmallVector<Attribute> flags;
  if (brgemmInfo.isVnni) {
    flags.push_back(xsmm::GemmFlagsAttr::get(rewriter.getContext(),
                                             xsmm::GemmFlags::VNNI_B));
  }
  // TODO: Support more than just COL_0 BCAST
  auto broadcastInput =
      isa<vector::BroadcastOp>(binaryOp->getOperand(0).getDefiningOp())
          ? binaryOp->getOperand(0).getDefiningOp()
          : binaryOp->getOperand(1).getDefiningOp();

  auto binaryFlags = xsmm::utils::getBinaryFlags(
      broadcastInput->getOperand(0).getDefiningOp()->getOperand(0).getType(),
      binaryOpResult->get().getType(), mlir::xsmm::utils::OperandPos::LHS);
  int binaryArg = 0;
  switch (*binaryFlags) {
  case mlir::xsmm::BinaryFlags::BCAST_COL_IN_0:
    binaryArg = 1;
    break;
  case mlir::xsmm::BinaryFlags::BCAST_COL_IN_1:
    binaryArg = 2;
    binaryFlags = mlir::xsmm::BinaryFlags::BCAST_COL_IN_0;
    break;
  default:
    return nullptr;
  }
  if (zeroOp) {
    flags.push_back(xsmm::GemmFlagsAttr::get(rewriter.getContext(),
                                             xsmm::GemmFlags::BETA_0));
  }
  ArrayAttr brgemmFlags = rewriter.getArrayAttr(flags);

  SmallVector<Value> invokeOperands;

  DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
      rewriter.getContext(),
      ArrayRef<int64_t>{m, n, k, lda, ldb, ldc, strideA, strideB});

  Value dispatched = rewriter.create<xsmm::FusedBrgemmDispatchOp>(
      loc, integer64, dims,
      xsmm::BinaryKindAttr::get(rewriter.getContext(), binaryKind),
      xsmm::UnaryKindAttr::get(rewriter.getContext(), unaryKind), brgemmFlags,
      rewriter.getArrayAttr(xsmm::UnaryFlagsAttr::get(rewriter.getContext(),
                                                      xsmm::UnaryFlags::NONE)),
      rewriter.getArrayAttr(
          xsmm::BinaryFlagsAttr::get(rewriter.getContext(), *binaryFlags)),
      dtype);

  Value batchDim = rewriter.create<arith::ConstantOp>(
      loc, integer64, rewriter.getIntegerAttr(integer64, batch));
  invokeOperands.push_back(dispatched);
  for (auto operand : inputs) {
    invokeOperands.push_back(operand->get());
  }
  invokeOperands.push_back(
      binaryOp->getOperand(binaryArg).getDefiningOp()->getOperand(0));
  invokeOperands.push_back(batchDim);
  auto brgemmOp =
      rewriter.create<xsmm::FusedBrgemmOp>(loc, dtype, invokeOperands);
  for (auto user : unaryOp->getResult(0).getUsers()) {
    rewriter.eraseOp(user);
  }
  rewriter.eraseOp(unaryOp);
  for (auto user : binaryOp->getResult(0).getUsers()) {
    rewriter.eraseOp(user);
  }
  rewriter.eraseOp(binaryOp);
  for (auto user : contractOp->getResult(0).getUsers()) {
    rewriter.eraseOp(user);
  }
  rewriter.eraseOp(zeroOp);
  rewriter.eraseOp(contractOp);
  return brgemmOp;
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

FailureOr<SmallVector<int64_t>> getVNNIStaticStrides(MemRefType valueType) {
  SmallVector<int64_t> strides;
  int64_t offset;
  SmallVector<int64_t> shape;
  for (int i = 0; i < valueType.getShape().size(); i++) {
    shape.push_back(valueType.getShape()[i]);
  }
  auto temp = shape[shape.size() - 1];
  shape[shape.size() - 1] = shape[shape.size() - 2];
  shape[shape.size() - 2] = temp;
  auto memrefType = MemRefType::get(shape, valueType.getElementType());
  if (failed(getStridesAndOffset(memrefType, strides, offset))) {
    return failure();
  }
  if (llvm::any_of(strides, [](int64_t stride) {
        return stride == ShapedType::kDynamic;
      })) {
    return failure();
  }
  return strides;
}

// Access matcher.
static FailureOr<BrgemmInfo>
checkAccess(PatternRewriter &rewriter, vector::ContractionOp contractOp,
            unsigned m, unsigned n, SmallVector<unsigned> kVector,
            std::optional<unsigned> batchPos, SmallVector<OpOperand *> inputs,
            ArrayRef<AffineMap> indexingMap) {
  OpOperand *operandA = inputs[0];
  OpOperand *operandB = inputs[1];
  OpOperand *operandC = inputs[2];

  unsigned k = 1;
  for (auto kItr : kVector)
    k *= kItr;
  auto checkStridesAndGetLda = [&](unsigned minorDim, unsigned majorDim,
                                   OpOperand *operand, AffineMap map,
                                   int operandIndex) -> FailureOr<int64_t> {
    auto minorDimPosInCodomain =
        getPosInCodomain(minorDim, operand, contractOp, map);
    auto majorDimPosInCodomain =
        getPosInCodomain(majorDim, operand, contractOp, map);
    if (!minorDimPosInCodomain || !majorDimPosInCodomain)
      return failure();
    auto dataType =
        xsmm::utils::getDataType(rewriter, inputs[0]->get().getType());
    FailureOr<SmallVector<int64_t>> stridesOnOperand;
    if (dataType == xsmm::DataTypeAttr::get(contractOp.getContext(),
                                            xsmm::DataType::BF16) &&
        operandIndex == 1) {
      stridesOnOperand =
          getVNNIStaticStrides(dyn_cast<MemRefType>(operand->get().getType()));
    } else {
      stridesOnOperand = utils::getStaticStrides(operand->get());
    }

    if (failed(stridesOnOperand) ||
        (dataType == xsmm::DataTypeAttr::get(contractOp.getContext(),
                                             xsmm::DataType::BF16) &&
         operandIndex == 0 &&
         (*stridesOnOperand)[*minorDimPosInCodomain] != 2) ||
        ((dataType != xsmm::DataTypeAttr::get(contractOp.getContext(),
                                              xsmm::DataType::BF16) &&
          (*stridesOnOperand)[*minorDimPosInCodomain] != 1))) {
      return failure();
    }
    if (dataType == xsmm::DataTypeAttr::get(contractOp.getContext(),
                                            xsmm::DataType::BF16) &&
        operandIndex == 1) {
      return (*stridesOnOperand)[*majorDimPosInCodomain + 1];
    } else
      return (*stridesOnOperand)[*majorDimPosInCodomain];
  };

  // A(m, k)
  auto lda = checkStridesAndGetLda(k, m, operandA, indexingMap[0], 0);
  if (failed(lda)) {
    return failure();
  }
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrge"
                             "mm] Strides on "
                             "A: OK\n");

  // B(k, n)
  auto ldb = checkStridesAndGetLda(n, k, operandB, indexingMap[1], 1);
  if (failed(ldb)) {
    return failure();
  }
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrge"
                             "mm] Strides on "
                             "B: OK\n");

  // C(m, n)
  auto ldc = checkStridesAndGetLda(n, m, operandC, indexingMap[2], 2);
  if (failed(ldc)) {
    return failure();
  }
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrge"
                             "mm] Strides on "
                             "C: OK\n");

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

  auto loopsK = 1;
  for (auto kItr : kVector)
    loopsK *= loops[kItr];

  BrgemmInfo info{loops[m], loops[n], loopsK,  batchVal, *lda,
                  *ldb,     *ldc,     strideA, strideB};
  return info;
}

// Check if the given
// generic is mappable to a
// brgemm xsmm op.
// - It is a contraction,
// with:
// -- 1 m and 1 n and 2 k
// dimensions.
// -- m appears on the LHS
// and OUT but not in RHS.
// -- n appears on the RHS
// and OUT but not in LHS.
// -- k and k' appear on the
// RHS and LHS but not OUT.
// -- the stride of the
// minor dimension for A, k
// is 1.
// -- the stride of the
// minor dimension for B, n
// is 1.
// -- the stride of the
// minor dimension for C, n
// is 1.
static FailureOr<BrgemmInfo>
isMappableToBrgemm(PatternRewriter &rewriter, vector::ContractionOp contractOp,
                   SmallVector<OpOperand *> &inputs,
                   SmallVector<OpOperand *> &output,
                   ArrayRef<AffineMap> indexingMap) {
  auto contractionDims =
      checkStructure(contractOp, inputs, output, indexingMap);
  if (failed(contractionDims)) {
    LLVM_DEBUG(llvm::dbgs() << "[isMappableToBr"
                               "gemm] Failed "
                               "on "
                               "checkStructure"
                               "\n");
    return failure();
  }
  unsigned m = contractionDims->m[0];
  unsigned n = contractionDims->n[0];
  SmallVector<unsigned> kVector;
  for (int i = 1; i < contractionDims->k.size(); i++)
    kVector.push_back(contractionDims->k[i]);
  std::optional<unsigned> batch;
  if (contractionDims->k.size() >= 2)
    batch = contractionDims->k.front();

  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrge"
                             "mm] Candidate "
                             "dims: "
                          << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrge"
                             "mm] m: "
                          << m << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrge"
                             "mm] n: "
                          << n << "\n");
  if (batch)
    LLVM_DEBUG(llvm::dbgs() << "[isMappableToBr"
                               "gemm] batch: "
                            << batch << "\n");
  else
    LLVM_DEBUG(llvm::dbgs() << "[isMappableToBr"
                               "gemm] no batch "
                               "dim\n");

  return checkAccess(rewriter, contractOp, m, n, kVector, batch, inputs,
                     indexingMap);
}

struct ConvertVectorTransposeToVnni2
    : public OpRewritePattern<vector::TransposeOp> {
  using OpRewritePattern<vector::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<std::function<bool(Operation * op)>> operations;
    operations.push_back(FuncType<vector::TransposeOp>);
    SmallVector<Operation *> opChain;
    if (!WithOps(&transposeOp->getParentOp()->getRegion(0),
                 transposeOp->getParentOp(), operations, opChain)) {
      return failure();
    }
    assert(opChain[0] == transposeOp);

    SmallVector<OpOperand *> inputs;
    SmallVector<std::function<bool(Operation * op)>> inputOperations;
    inputOperations.push_back(FuncType<vector::TransferReadOp>);
    if (!WithInputs(rewriter, transposeOp, inputOperations, inputs)) {
      return failure();
    }
    SmallVector<OpOperand *> outputs;
    if (!WithOutput(transposeOp, FuncType<vector::TransferWriteOp>, outputs)) {
      return failure();
    }

    Value source = inputs[0]->get();
    MemRefType outType = cast<MemRefType>(outputs[0]->get().getType());
    MemRefType sourceType = cast<MemRefType>(source.getType());
    if (!outType.hasStaticShape() || !sourceType.hasStaticShape() ||
        !vnni::utils::isInVnniLayout(vnni::utils::VnniOperandRank::TRANSPOSE,
                                     outType)) {
      return failure();
    }

    memref::ExpandShapeOp expandShapeOp =
        dyn_cast<memref::ExpandShapeOp>(source.getDefiningOp());
    if (!expandShapeOp || expandShapeOp.getSrcType().getRank() != 2)
      return failure();

    source = expandShapeOp.getSrc();
    xsmm::UnaryInfo unaryInfo;
    unaryInfo.m = expandShapeOp.getSrcType().getShape()[0];
    unaryInfo.n = expandShapeOp.getSrcType().getShape()[1];
    auto stridesOnInput = mlir::utils::getStaticStrides(source);
    if (failed(stridesOnInput) || stridesOnInput->back() != 1)
      return failure();
    unaryInfo.ldi = stridesOnInput->front();
    auto stridesOnOutput = mlir::utils::getStaticStrides(outputs[0]->get());
    if (failed(stridesOnOutput) || stridesOnOutput->back() != 1)
      return failure();
    // Ajust ldo based on the VNNI factor.
    unaryInfo.ldo =
        stridesOnOutput->front() /
        *vnni::utils::getVnniBlockingFactor(outputs[0]->get().getType());
    auto flags = rewriter.getArrayAttr(xsmm::UnaryFlagsAttr::get(
        rewriter.getContext(), xsmm::UnaryFlags::NONE));
    xsmm::UnaryKindAttr kind =
        xsmm::UnaryKindAttr::get(rewriter.getContext(), xsmm::UnaryKind::VNNI2);
    Location loc = transposeOp->getLoc();
    IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
    DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
        rewriter.getContext(), ArrayRef<int64_t>{unaryInfo.m, unaryInfo.n,
                                                 unaryInfo.ldi, unaryInfo.ldo});
    auto dtype = xsmm::utils::getDataType(rewriter, expandShapeOp.getSrcType());
    Value dispatched = rewriter.create<xsmm::UnaryDispatchOp>(
        loc, integer64, kind, dims, flags, dtype);
    SmallVector<Value> invokeOperands;
    invokeOperands.push_back(dispatched);
    invokeOperands.push_back(source);
    invokeOperands.push_back(outputs[0]->get());

    rewriter.create<xsmm::UnaryOp>(transposeOp.getLoc(), dtype, kind,
                                   invokeOperands);

    for (auto user : transposeOp->getUsers())
      rewriter.eraseOp(user);
    rewriter.eraseOp(transposeOp);
    return success();
  }
};

// Convert vector.contract
// to a XSMM brgemm op.
struct ConvertVectorContractToBatchReduceMatmul
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {

    SmallVector<std::function<bool(Operation * op)>> operations;
    operations.push_back(FuncType<vector::ContractionOp>);
    SmallVector<Operation *> opChain;
    if (!WithOps(&contractOp->getParentOp()->getRegion(0),
                 contractOp->getParentOp(), operations, opChain)) {
      return failure();
    }
    assert(opChain[0] == contractOp);

    SmallVector<std::function<bool(Operation * op)>> inputOperations;
    inputOperations.push_back(FuncType<vector::TransferReadOp>);
    inputOperations.push_back(FuncType<vector::TransferReadOp>);
    inputOperations.push_back(FuncType<vector::TransferReadOp>);
    SmallVector<OpOperand *> inputs;

    if (!WithInputs(rewriter, contractOp, inputOperations, inputs)) {
      return failure();
    }

    if (contractOp.getKind() != CombiningKind::ADD) {
      return failure();
    }
    auto indexingMaps = contractOp.getIndexingMaps();
    if (indexingMaps.size() != 3)
      return failure();

    auto dataType =
        xsmm::utils::getDataType(rewriter, inputs[0]->get().getType());
    SmallVector<AffineMap> affineMaps;
    auto iteratorTypes = contractOp.getIteratorTypes();
    if (dataType ==
        xsmm::DataTypeAttr::get(rewriter.getContext(), xsmm::DataType::BF16)) {
      if (iteratorTypes.size() != 5) {
        return failure();
      }
      size_t size = iteratorTypes.size() - 1;
      bool match = vector::isReductionIterator(iteratorTypes[size]) &&
                   vector::isParallelIterator(iteratorTypes[size - 1]) &&
                   vector::isParallelIterator(iteratorTypes[size - 2]) &&
                   vector::isReductionIterator(iteratorTypes[size - 3]) &&
                   vector::isReductionIterator(iteratorTypes[size - 4]);
      if (!match) {
        return failure();
      }
      AffineExpr r0, r1, p3, p4, r2;
      using MapList = ArrayRef<ArrayRef<AffineExpr>>;
      auto infer = [&](MapList m, PatternRewriter &rewriter) {
        return rewriter.getAffineMapArrayAttr(
            AffineMap::inferFromExprList(m, contractOp.getContext()));
      };

      bindDims(contractOp.getContext(), r0, r1, p3, p4, r2);
      auto expectedMaps =
          infer({{r0, p3, r2, r1}, {r0, r2, p4, r1}, {p3, p4}}, rewriter);

      if (indexingMaps != expectedMaps) {
        return failure();
      }
      SmallVector<ArrayRef<AffineExpr>> map = {
          {r0, p3, r2, r1}, {r0, r2, p4, r1}, {p3, p4}};
      affineMaps = AffineMap::inferFromExprList(map, contractOp.getContext());
    } else {
      if (iteratorTypes.size() != 4) {
        return failure();
      }
      size_t size = iteratorTypes.size() - 1;
      bool match = vector::isReductionIterator(iteratorTypes[size]) &&
                   vector::isParallelIterator(iteratorTypes[size - 1]) &&
                   vector::isParallelIterator(iteratorTypes[size - 2]) &&
                   vector::isReductionIterator(iteratorTypes[size - 3]);
      if (!match) {
        return failure();
      }
      AffineExpr r1, p3, p4, r2;
      using MapList = ArrayRef<ArrayRef<AffineExpr>>;
      auto infer = [&](MapList m, PatternRewriter &rewriter) {
        return rewriter.getAffineMapArrayAttr(
            AffineMap::inferFromExprList(m, contractOp.getContext()));
      };

      bindDims(contractOp.getContext(), r1, p3, p4, r2);
      auto expectedMaps =
          infer({{r1, p3, r2}, {r1, r2, p4}, {p3, p4}}, rewriter);

      if (indexingMaps != expectedMaps) {
        return failure();
      }
      SmallVector<ArrayRef<AffineExpr>> map = {
          {r1, p3, r2}, {r1, r2, p4}, {p3, p4}};
      affineMaps = AffineMap::inferFromExprList(map, contractOp.getContext());
    }
    SmallVector<OpOperand *> outputs;

    operations.clear();
    operations.push_back(FuncType<arith::AddFOp>);
    opChain.clear();
    if (WithOps(&contractOp->getParentOp()->getRegion(0),
                contractOp->getParentOp(), operations, opChain)) {
      auto biasAdd = opChain[0];
      assert(isa<arith::AddFOp>(biasAdd));
      SmallVector<std::function<bool(Operation * op)>> inputOperations;
      inputOperations.push_back(FuncType<vector::BroadcastOp>);
      inputOperations.push_back(FuncType<vector::TransferReadOp>);
      SmallVector<OpOperand *> addInputs;
      if (WithInputs(rewriter, biasAdd, inputOperations, addInputs)) {
        SmallVector<OpOperand *> addOutputs;
        if (WithOutput(biasAdd, FuncType<vector::TransferWriteOp>,
                       addOutputs)) {
          operations.clear();
          operations.push_back(FuncType<arith::MaximumFOp>);
          opChain.clear();
          if (WithOps(&contractOp->getParentOp()->getRegion(0),
                      contractOp->getParentOp(), operations, opChain)) {
            auto maxF = opChain[0];
            assert(isa<arith::MaximumFOp>(maxF));
            SmallVector<OpOperand *> maxfInputs;
            SmallVector<std::function<bool(Operation * op)>> maxFOperations;
            maxFOperations.push_back(FuncType<vector::TransferReadOp>);
            if (WithInputs(rewriter, maxF, maxFOperations, maxfInputs)) {
              SmallVector<OpOperand *> maxfOutputs;
              if (WithOutput(maxF, FuncType<vector::TransferWriteOp>,
                             maxfOutputs)) {
                if (WithOutput(contractOp, FuncType<vector::TransferWriteOp>,
                               outputs)) {

                  auto fusedBrgemmInfo = isMappableToBrgemm(
                      rewriter, contractOp, inputs, outputs, affineMaps);
                  if (failed(fusedBrgemmInfo)) {
                    return failure();
                  }
                  vector::TransferWriteOp zeroOp;
                  WithZeroInit(inputs[2], zeroOp);
                  replaceOpWithFusedBrgemmOp(
                      rewriter, contractOp, *fusedBrgemmInfo, inputs,
                      xsmm::UnaryKind::RELU, xsmm::BinaryKind::ADD,
                      addInputs[0], maxF, biasAdd, maxfOutputs[0], zeroOp);
                }
              }
            }
          }
        }
      }
    } else {
      if (WithOutput(contractOp, FuncType<vector::TransferWriteOp>, outputs)) {
        auto brgemmInfo = isMappableToBrgemm(rewriter, contractOp, inputs,
                                             outputs, affineMaps);
        if (failed(brgemmInfo)) {
          return failure();
        }
        replaceOpWithGemmLikeOp(rewriter, contractOp, *brgemmInfo, inputs);
      }
    }
    return success();
  }
};

void populateVectorToXsmmPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertVectorContractToBatchReduceMatmul,
               ConvertVectorTransposeToVnni2>(patterns.getContext());
}

void ConvertVectorToXsmm::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateVectorToXsmmPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

} // namespace tpp
} // namespace mlir
