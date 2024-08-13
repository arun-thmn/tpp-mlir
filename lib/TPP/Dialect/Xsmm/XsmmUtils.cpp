//===- XsmmUtils.cpp ---------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Xsmm/XsmmUtils.h"
#include "TPP/Dialect/Xsmm/XsmmOps.h"
#include "TPP/Transforms/Utils/ValueUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Compiler.h"
#define DEBUG_TYPE "xsmm-utils"

using namespace mlir;
using namespace mlir::linalg;
using namespace structured_match;

namespace mlir {
namespace xsmm {
namespace utils {

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

// Return the position of `dim` in the codomain of `operand`.
static std::optional<unsigned>
getPosInCodomain(unsigned dim, OpOperand *operand,
                 vector::ContractionOp contractOp, AffineMap map) {
  return map.getResultPosition(getAffineDimExpr(dim, contractOp.getContext()));
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

static FailureOr<SmallVector<int64_t>>
getVNNIStaticStrides(MemRefType valueType) {
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
static FailureOr<xsmm::BrgemmInfo>
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
      stridesOnOperand = ::mlir::utils::getStaticStrides(operand->get());
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
    auto stridesOnA = ::mlir::utils::getStaticStrides(operandA->get());
    strideA = (*stridesOnA)[*batchPosCodomainA];

    auto batchPosCodomainB = getPosInCodomain(batchPos.value(), operandB,
                                              contractOp, indexingMap[1]);
    auto stridesOnB = ::mlir::utils::getStaticStrides(operandB->get());
    strideB = (*stridesOnB)[*batchPosCodomainB];
  }

  auto loops = computeStaticLoopSizes(contractOp, indexingMap);
  int64_t batchVal = (batchPos) ? loops[batchPos.value()] : 0;

  auto loopsK = 1;
  for (auto kItr : kVector)
    loopsK *= loops[kItr];

  xsmm::BrgemmInfo info{loops[m], loops[n], loopsK,  batchVal, *lda,
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
FailureOr<BrgemmInfo> isMappableToBrgemm(PatternRewriter &rewriter,
                                         vector::ContractionOp contractOp,
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

DataTypeAttr getDataType(RewriterBase &rewriter, Type type) {
  auto elemType = getElementTypeOrSelf(type);
  if (elemType.isBF16())
    return xsmm::DataTypeAttr::get(rewriter.getContext(), xsmm::DataType::BF16);
  return xsmm::DataTypeAttr::get(rewriter.getContext(), xsmm::DataType::F32);
}

void replaceOpWithUnary(RewriterBase &rewriter, Operation *operation,
                        ArrayRef<Value> operands, UnaryInfo unaryInfo,
                        ArrayAttr flags, xsmm::UnaryKindAttr kind) {
  Location loc = operation->getLoc();
  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
      rewriter.getContext(), ArrayRef<int64_t>{unaryInfo.m, unaryInfo.n,
                                               unaryInfo.ldi, unaryInfo.ldo});
  auto dtype = xsmm::utils::getDataType(rewriter, operands.back().getType());
  Value dispatched = rewriter.create<xsmm::UnaryDispatchOp>(
      loc, integer64, kind, dims, flags, dtype);
  SmallVector<Value> invokeOperands;
  invokeOperands.push_back(dispatched);
  invokeOperands.append(operands.begin(), operands.end());
  rewriter.replaceOpWithNewOp<xsmm::UnaryOp>(operation, dtype, kind,
                                             invokeOperands);
}

FailureOr<UnaryInfo> getUnaryInfo(Value input, Value output,
                                  UnaryFlags inputFlag) {
  Type outputType = output.getType();

  assert(isa<ShapedType>(outputType));
  auto outputShapedType = cast<ShapedType>(outputType);
  if (outputShapedType.getRank() != 2 || !outputShapedType.hasStaticShape() ||
      !isa<FloatType>(outputShapedType.getElementType())) {
    return failure();
  }

  UnaryInfo unaryInfo;
  unaryInfo.m = outputShapedType.getShape()[0];
  unaryInfo.n = outputShapedType.getShape()[1];

  int64_t ldi = 1;
  if (ShapedType inputShapedType = dyn_cast<ShapedType>(input.getType())) {
    auto stridesOnInput = mlir::utils::getStaticStrides(input);
    if (failed(stridesOnInput) || stridesOnInput->back() != 1 ||
        !inputShapedType.hasStaticShape()) {
      return failure();
    }

    // If we are broascasting a row into cols, the leading
    // dimension is 1, same for scalar broadcast.
    if (inputFlag == UnaryFlags::BCAST_ROW ||
        inputFlag == UnaryFlags::BCAST_SCALAR) {
      ldi = 1;
    }
    // If we are broascasting a col into rows, the leading
    // dimension is the size of the tensor.
    else if (inputFlag == UnaryFlags::BCAST_COL) {
      ldi = inputShapedType.getShape().back();
    } else {
      ldi = stridesOnInput->front();
    }
  }
  auto stridesOnOutput = mlir::utils::getStaticStrides(output);
  if (failed(stridesOnOutput) || stridesOnOutput->back() != 1)
    return failure();

  unaryInfo.ldi = ldi;
  unaryInfo.ldo = stridesOnOutput->front();
  return unaryInfo;
}

FailureOr<BinaryInfo> getBinaryInfo(Value lhs, BinaryFlags lhsFlag, Value rhs,
                                    BinaryFlags rhsFlag, Value output) {
  Type outputType = output.getType();

  assert(isa<ShapedType>(outputType));
  auto outputShapedType = cast<ShapedType>(outputType);
  if (outputShapedType.getRank() != 2 || !outputShapedType.hasStaticShape() ||
      !isa<FloatType>(outputShapedType.getElementType())) {
    return failure();
  }

  BinaryInfo binaryInfo;
  binaryInfo.m = outputShapedType.getShape()[0];
  binaryInfo.n = outputShapedType.getShape()[1];

  int64_t ldiLhs = 1;
  if (ShapedType lhsShapedType = dyn_cast<ShapedType>(lhs.getType())) {
    auto stridesOnLhs = mlir::utils::getStaticStrides(lhs);
    if (failed(stridesOnLhs) || stridesOnLhs->back() != 1 ||
        !lhsShapedType.hasStaticShape()) {
      return failure();
    }

    if (lhsFlag == BinaryFlags::BCAST_SCALAR_IN_0 ||
        lhsFlag == BinaryFlags::BCAST_ROW_IN_0) {
      ldiLhs = 1;
    } else if (lhsFlag == BinaryFlags::BCAST_COL_IN_0) {
      ldiLhs = lhsShapedType.getShape().back();
    } else {
      ldiLhs = stridesOnLhs->front();
    }
  }

  int64_t ldiRhs = 1;
  if (ShapedType rhsShapedType = dyn_cast<ShapedType>(rhs.getType())) {
    auto stridesOnRhs = mlir::utils::getStaticStrides(rhs);
    if (failed(stridesOnRhs) || stridesOnRhs->back() != 1 ||
        !rhsShapedType.hasStaticShape()) {
      return failure();
    }

    if (rhsFlag == BinaryFlags::BCAST_SCALAR_IN_1 ||
        rhsFlag == BinaryFlags::BCAST_ROW_IN_1) {
      ldiRhs = 1;
    } else if (rhsFlag == BinaryFlags::BCAST_COL_IN_1) {
      ldiRhs = rhsShapedType.getShape().back();
    } else {
      ldiRhs = stridesOnRhs->front();
    }
  }

  binaryInfo.ldiLhs = ldiLhs;
  binaryInfo.ldiRhs = ldiRhs;

  auto stridesOnOutput = mlir::utils::getStaticStrides(output);
  if (failed(stridesOnOutput) || stridesOnOutput->back() != 1)
    return failure();
  binaryInfo.ldo = stridesOnOutput->front();
  return binaryInfo;
}

// Examples:
// If lower=[c], higher=[a, b, c], [c] reshaped into [1, 1, c].
// If lower=[b, c], higher=[a, b, c], [b, c] reshaped into [1, b, c].
// If lower=[a], higher=[a, a], [a] reshaped into [1, a].
// If lower=[a], target=[a, b, a], [a] reshaped into [1, 1, a].
// If lower=[], target=[a, b, c], [] reshaped into [1, 1, 1].
static void
computeBcastShapeInput(ArrayRef<int64_t> higherRankShape,
                       ArrayRef<int64_t> lowerRankShape,
                       SmallVectorImpl<int64_t> &reshapeOutputShape) {
  // Initialize new shapes with [1] * higherRank.
  int64_t higherRank = higherRankShape.size();
  int64_t lowerRank = lowerRankShape.size();

  reshapeOutputShape.assign(higherRank, 1);

  int64_t higherRankDim;
  int64_t lowerRankDim;

  for (int64_t i = higherRank - 1, j = lowerRank - 1; i >= 0 && j >= 0;
       i--, j--) {
    higherRankDim = higherRankShape[i];
    lowerRankDim = lowerRankShape[j];

    if (lowerRankDim == 1 && higherRankDim > 1)
      reshapeOutputShape[i] = 1;
    else if ((lowerRankDim > 1 && higherRankDim == 1) ||
             (lowerRankDim == higherRankDim))
      reshapeOutputShape[i] = lowerRankDim;
    else if (higherRankDim != lowerRankDim)
      assert(false && "bCast semantics for identity op broken");
  }
}

FailureOr<UnaryFlags> getUnaryFlags(Type inputType, Type outputType) {
  assert(isa<ShapedType>(outputType) && "expect shaped type on output");
  assert(cast<ShapedType>(outputType).getRank() == 2 &&
         "expect rank 2 on output");

  if (!isa<ShapedType>(inputType) ||
      cast<ShapedType>(inputType).getRank() == 0) {
    return xsmm::UnaryFlags::BCAST_SCALAR;
  }

  ArrayRef<int64_t> shapeOutput = cast<ShapedType>(outputType).getShape();
  ArrayRef<int64_t> shapeInput = cast<ShapedType>(inputType).getShape();
  assert(shapeOutput.size() >= shapeInput.size() &&
         "output rank must be >= input rank");
  SmallVector<int64_t> bShapeInput;
  computeBcastShapeInput(shapeOutput, shapeInput, bShapeInput);
  assert(shapeOutput.size() == bShapeInput.size());
  shapeInput = bShapeInput;

  // Same shape for input and output, no bcast.
  if (shapeInput == shapeOutput)
    return xsmm::UnaryFlags::NONE;

  // Input is a memref but it is all ones, bcast = scalar.
  auto isOne = [](int64_t val) { return val == 1; };
  if (llvm::all_of(shapeInput, isOne))
    return xsmm::UnaryFlags::BCAST_SCALAR;

  if (shapeInput[1] == 1 && shapeOutput[1] > 1)
    return xsmm::UnaryFlags::BCAST_ROW;

  if (shapeInput[0] == 1 && shapeOutput[0] > 1)
    return xsmm::UnaryFlags::BCAST_COL;

  return failure();
}

FailureOr<BinaryFlags> getBinaryFlags(Type operandType, Type outputType,
                                      OperandPos operandNumber) {
  assert(isa<ShapedType>(outputType) && "expect shaped type on output");
  assert(cast<ShapedType>(outputType).getRank() == 2 &&
         "expect rank 2 on output");

  if (!isa<ShapedType>(operandType) ||
      cast<ShapedType>(operandType).getRank() == 0) {
    if (operandNumber == OperandPos::LHS)
      return xsmm::BinaryFlags::BCAST_SCALAR_IN_0;
    return xsmm::BinaryFlags::BCAST_SCALAR_IN_1;
  }

  enum class BCastType { NONE = 0, SCALAR, ROW, COL };
  auto shapeOutput = cast<MemRefType>(outputType).getShape();
  auto shapeOperand = cast<MemRefType>(operandType).getShape();
  assert(shapeOutput.size() >= shapeOperand.size() &&
         "Output rank must be >= operand rank");
  SmallVector<int64_t> bOperandShape;
  computeBcastShapeInput(shapeOutput, shapeOperand, bOperandShape);
  assert(shapeOutput.size() == bOperandShape.size());
  assert(shapeOutput.size() == 2);

  auto getBCastEnum = [](BCastType bCastType,
                         OperandPos operandPos) -> xsmm::BinaryFlags {
    switch (bCastType) {
    case BCastType::NONE:
      return xsmm::BinaryFlags::NONE;
    case BCastType::SCALAR:
      if (operandPos == OperandPos::LHS)
        return xsmm::BinaryFlags::BCAST_SCALAR_IN_0;
      else
        return xsmm::BinaryFlags::BCAST_SCALAR_IN_1;
    case BCastType::ROW:
      if (operandPos == OperandPos::LHS)
        return xsmm::BinaryFlags::BCAST_ROW_IN_0;
      else
        return xsmm::BinaryFlags::BCAST_ROW_IN_1;
    case BCastType::COL:
      if (operandPos == OperandPos::LHS)
        return xsmm::BinaryFlags::BCAST_COL_IN_0;
      else
        return xsmm::BinaryFlags::BCAST_COL_IN_1;
    }
    assert(false && "unrechable");
    abort();
  };

  if (bOperandShape == shapeOutput)
    return getBCastEnum(BCastType::NONE, operandNumber);

  auto isOne = [](int64_t val) { return val == 1; };
  if (llvm::all_of(bOperandShape, isOne))
    return getBCastEnum(BCastType::SCALAR, operandNumber);

  if (bOperandShape[1] == 1 && shapeOutput[1] > 1)
    return getBCastEnum(BCastType::ROW, operandNumber);

  if (bOperandShape[0] == 1 && shapeOutput[0] > 1)
    return getBCastEnum(BCastType::COL, operandNumber);

  return failure();
}

FailureOr<FusedMatch> getFusedBrgemmSequenceFromProducer(Operation *op) {
  // The loop is in reverse order, so we deduplicate the list making sure we
  // only have one type of each
  SmallVector<Operation *, 4> chain;
  Operation *prev = nullptr;
  for (auto *user : op->getUsers()) {
    // Deduplicate, only take each operation once
    if (dyn_cast<func::ReturnOp>(user) || user == prev)
      continue;
    chain.push_back(user);
    prev = user;

    // BRGEMM is the last one, we can stop looking
    if (auto brgemmOp = (dyn_cast<xsmm::BrgemmOp>(user))) {
      // Make sure the BRGEMM outputs to the chain value
      // (it could be one of BRGEMM's inputs in the chain)
      if (brgemmOp.getOperand(3).getDefiningOp() != op)
        return failure();
      continue;
    }

    // Make sure this is a chain, ie. at least once in inputs and outputs
    int numUses = std::count(user->getOperands().begin(),
                             user->getOperands().end(), op->getResult(0));
    // At least one input and the last operand (output) is the same buffer
    if (((dyn_cast<xsmm::UnaryOp>(user) &&
          dyn_cast<xsmm::UnaryOp>(user).getCallee() != UnaryKind::ZERO) &&
         numUses < 2) ||
        user->getOperands()[user->getOperands().size() - 1] != op->getResult(0))
      return failure();
  }
  // We don't know how to fuse more than two tail ops after and a zero op before
  // BRGEMM
  if (chain.size() > 4)
    return failure();
  if (!(isa<xsmm::BrgemmOp>(chain[0]) ||
        (dyn_cast<xsmm::UnaryOp>(chain[0]) &&
         dyn_cast<xsmm::UnaryOp>(chain[0]).getCallee() == UnaryKind::ZERO)))
    // List is in reverse order, put the brgemm or zero at the top
    std::reverse(chain.begin(), chain.end());

  // If we haven't found a BRGEMM or zero, this are not the droids we're looking
  // for
  assert((isa<xsmm::BrgemmOp>(chain[0]) ||
          (dyn_cast<xsmm::UnaryOp>(chain[0]) &&
           dyn_cast<xsmm::UnaryOp>(chain[0]).getCallee() == UnaryKind::ZERO &&
           isa<xsmm::BrgemmOp>(chain[1]))) &&
         "First op must be brgemm or zero");

  // Now, we're sure we have a chain, but not yet if it has the right types
  // and in the right order: (ZER0) -> BRGEMM -> BINARY -> UNARY
  // Allowed patterns are:
  //  - (ZERO) + GEMM + BINARY
  //  - (ZERO)+ GEMM + UNARY
  //  - (ZERO) + GEMM + BINARY + UNARY
  xsmm::FusedMatch fusedMatch;
  for (auto *user : chain) {
    if (auto unaryOp = dyn_cast<xsmm::UnaryOp>(user)) {
      if (dyn_cast<xsmm::UnaryOp>(user).getCallee() == UnaryKind::ZERO) {
        fusedMatch.zeroOp = unaryOp;
        continue;
      }
    }
    if (auto brgemmOp = (dyn_cast<xsmm::BrgemmOp>(user))) {
      // We only accept one of each
      if (fusedMatch.brgemmOp)
        return failure();

      fusedMatch.brgemmOp = brgemmOp;
      continue;
    }

    if (auto binOp = (dyn_cast<xsmm::BinaryOp>(user))) {
      // We only accept one of each
      if (fusedMatch.binaryOp)
        return failure();

      // We cannot accept binary *after* unary
      if (fusedMatch.unaryOp)
        return failure();

      // For now we only support ADD as binary
      if (binOp.getCallee() != BinaryKind::ADD)
        return failure();

      // Make sure the op is new or the same as before
      fusedMatch.binaryOp = binOp;
      fusedMatch.binaryKind = binOp.getCallee();
      continue;
    }

    if (auto unOp = dyn_cast<xsmm::UnaryOp>(user)) {
      // We only accept one of each
      if (fusedMatch.unaryOp)
        return failure();

      // Binary op may have come earlier, we don't know
      // We have already made sure it didn't come before this
      // unary in the binary check above

      // For now we only support RELU as unary
      if (unOp.getCallee() != UnaryKind::RELU)
        return failure();

      // Make sure the op is new or the same as before
      fusedMatch.unaryOp = unOp;
      fusedMatch.unaryKind = unOp.getCallee();
      continue;
    }

    // If found anything else in the users, bail
    return failure();
  }

  return fusedMatch;
}

FailureOr<int64_t> getLeadingDim(Type type, size_t pos) {
  // Not shaped type, the leading dimension is the single scalar.
  auto memref = dyn_cast<MemRefType>(type);
  if (!memref)
    return 1;
  // For 1d memref we cannot use the stride as leading dimension, but the
  // leading dimension is the dimension itself.
  if (memref.getRank() == 1)
    return memref.getShape()[0];

  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(getStridesAndOffset(memref, strides, offset)))
    return failure();
  // fail if the strides are non-constant
  if (llvm::any_of(strides, [](int64_t stride) {
        return stride == ShapedType::kDynamic;
      }))
    return failure();
  return strides[pos];
}

template <typename DispatchOpTy>
FailureOr<SmallVector<Attribute>> getBrgemmFlags(PatternRewriter &rewriter,
                                                 DispatchOpTy dispatchOpTy,
                                                 bool returnNone) {
  SmallVector<Attribute> attributes;
  auto flags = dispatchOpTy.getFlags();
  for (auto flagItr : flags) {
    if (flagItr == xsmm::GemmFlagsAttr::get(rewriter.getContext(),
                                            xsmm::GemmFlags::NONE)) {
      if (returnNone) {
        attributes.push_back(xsmm::GemmFlagsAttr::get(rewriter.getContext(),
                                                      xsmm::GemmFlags::NONE));
        return attributes;
      } else {
        return failure();
      }
    }
    attributes.push_back(flagItr);
  }

  if (attributes.empty())
    attributes.push_back(
        xsmm::GemmFlagsAttr::get(rewriter.getContext(), xsmm::GemmFlags::NONE));
  return attributes;
}

template FailureOr<SmallVector<Attribute>>
getBrgemmFlags<xsmm::BrgemmDispatchOp>(PatternRewriter &rewriter,
                                       xsmm::BrgemmDispatchOp dispatchOpTy,
                                       bool returnNone);

template FailureOr<SmallVector<Attribute>>
getBrgemmFlags<xsmm::FusedBrgemmDispatchOp>(
    PatternRewriter &rewriter, xsmm::FusedBrgemmDispatchOp dispatchOpTy,
    bool returnNone);

bool WithInputs(PatternRewriter &rewriter, Operation *op,
                SmallVector<std::function<bool(Operation *op)>> operations,
                SmallVector<OpOperand *> &inputs,
                SmallVector<Operation *> &opChain) {
  for (int i = 0; i < operations.size(); i++) {
    auto input = op->getOperand(i);
    if (!operations[i](input.getDefiningOp()))
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
    opChain.push_back(input.getDefiningOp());
  }
  return true;
}

bool WithOutput(Operation *op, std::function<bool(Operation *op)> operation,
                SmallVector<OpOperand *> &output,
                SmallVector<Operation *> &opChain) {
  // Check on the inner chain of operations in the right order.
  // Make sure all operands are used and chained
  for (auto use : op->getResult(0).getUsers()) {
    if (use != op && operation(use)) {
      assert(isa<memref::SubViewOp>(use->getOperand(1).getDefiningOp()));
      output.push_back(&use->getOpOperand(1));
      opChain.push_back(use);
      return true;
    }
  }
  return false;
}

bool WithOps(Region *region, Operation *op,
             SmallVector<std::function<bool(Operation *op)>> operations,
             SmallVector<Operation *> &opChain) {
  // Basic checks
  if (!isa<scf::ForallOp>(op) && !isa<scf::ForOp>(op) &&
      !isa<scf::ParallelOp>(op))
    return false;
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

Operation *WithZeroInit(OpOperand *input,
                        vector::TransferWriteOp &transferWriteOp) {
  Operation *rootOp = nullptr;
  for (auto user : input->get().getUsers()) {
    if (isa<vector::TransferWriteOp>(user) &&
        ::mlir::utils::isValConstZero(
            dyn_cast<vector::TransferWriteOp>(user).getOperand(0))) {
      transferWriteOp = dyn_cast<vector::TransferWriteOp>(user);
      rootOp = transferWriteOp;
      break;
    }
  }
  if (rootOp == nullptr)
    return nullptr;
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

} // namespace utils
} // namespace xsmm
} // namespace mlir
