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

static std::optional<Operation *>
replaceOpWithGemmLikeOp(RewriterBase &rewriter,
                        vector::ContractionOp contractOp,
                        xsmm::BrgemmInfo brgemmInfo,
                        SmallVector<OpOperand *> inputs, Operation *zeroOp) {
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
  if (zeroOp) {
    flags.push_back(xsmm::GemmFlagsAttr::get(rewriter.getContext(),
                                             xsmm::GemmFlags::BETA_0));
  }
  ArrayAttr brgemmFlags = rewriter.getArrayAttr(flags);
  SmallVector<Value> invokeOperands;

  if (batch != 0) {
    DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
        rewriter.getContext(),
        ArrayRef<int64_t>{m, n, k, lda, ldb, ldc, strideA, strideB});
    Value dispatched = rewriter.create<xsmm::BrgemmDispatchOp>(
        loc, integer64, dims, brgemmFlags, dtype);
    Value batchDim = rewriter.create<arith::ConstantOp>(
        loc, integer64, rewriter.getIntegerAttr(integer64, batch));
    invokeOperands.push_back(dispatched);
    for (auto operand : inputs) {
      invokeOperands.push_back(operand->get());
    }
    invokeOperands.push_back(batchDim);
    auto brgemmOp = rewriter.create<xsmm::BrgemmOp>(loc, dtype, invokeOperands);
    if (!contractOp->use_empty()) {
      for (auto user = contractOp->user_begin();
           user != contractOp->user_end() && !contractOp->use_empty(); user++) {
        auto contractUser = *user;
        if (contractUser->use_empty()) {
          rewriter.eraseOp(contractUser);
        }
      }
    }
    if (contractOp->use_empty())
      rewriter.eraseOp(contractOp);
    return brgemmOp;
  } else {
    DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
        rewriter.getContext(), ArrayRef<int64_t>{m, n, k, lda, ldb, ldc});
    Value dispatched = rewriter.create<xsmm::GemmDispatchOp>(
        loc, integer64, dims, brgemmFlags, dtype);
    invokeOperands.push_back(dispatched);
    for (auto operand : inputs) {
      invokeOperands.push_back(operand->get());
    }
    auto gemmOp = rewriter.create<xsmm::GemmOp>(loc, dtype, invokeOperands);
    if (!contractOp->use_empty()) {

      for (auto user = contractOp->user_begin();
           user != contractOp->user_end() && !contractOp->use_empty(); user++) {
        auto contractUser = *user;
        if (contractUser->use_empty()) {
          rewriter.eraseOp(contractUser);
        }
      }
    }
    if (zeroOp) {
      rewriter.eraseOp(zeroOp);
    }
    if (contractOp->use_empty())
      rewriter.eraseOp(contractOp);
    return gemmOp;
  }
}

static std::optional<Operation *> replaceOpWithFusedBrgemmOp(
    RewriterBase &rewriter, vector::ContractionOp contractOp,
    xsmm::BrgemmInfo brgemmInfo, SmallVector<OpOperand *> inputs,
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
  if (!unaryOp->use_empty()) {
    for (auto user = unaryOp->user_begin();
         user != unaryOp->user_end() && !unaryOp->use_empty(); user++) {
      auto use = *user;
      rewriter.eraseOp(use);
    }
  }
  if (unaryOp->use_empty())
    rewriter.eraseOp(unaryOp);
  if (!binaryOp->use_empty()) {
    for (auto user = binaryOp->user_begin();
         user != binaryOp->user_end() && !binaryOp->use_empty(); user++) {
      auto use = *user;
      rewriter.eraseOp(use);
    }
  }
  if (binaryOp->use_empty())
    rewriter.eraseOp(binaryOp);
  if (!contractOp->use_empty()) {
    for (auto user = contractOp->user_begin();
         user != contractOp->user_end() && !contractOp->use_empty(); user++) {
      auto use = *user;
      rewriter.eraseOp(use);
    }
  }
  if (zeroOp) {
    rewriter.eraseOp(zeroOp);
  }
  if (contractOp->use_empty())
    rewriter.eraseOp(contractOp);
  return brgemmOp;
}

struct ConvertVectorTransposeToVnni2
    : public OpRewritePattern<vector::TransposeOp> {
  using OpRewritePattern<vector::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<std::function<bool(Operation * op)>> operations;
    operations.push_back(xsmm::FuncType<vector::TransposeOp>);
    SmallVector<Operation *> opChain;
    if (!xsmm::utils::WithOps(&transposeOp->getParentOp()->getRegion(0),
                              transposeOp->getParentOp(), operations,
                              opChain)) {
      return failure();
    }
    assert(opChain[0] == transposeOp);

    SmallVector<OpOperand *> inputs;
    SmallVector<std::function<bool(Operation * op)>> inputOperations;
    inputOperations.push_back(xsmm::FuncType<vector::TransferReadOp>);
    SmallVector<Operation *> inputOpChain;
    if (!xsmm::utils::WithInputs(rewriter, transposeOp, inputOperations, inputs,
                                 inputOpChain)) {
      return failure();
    }
    SmallVector<OpOperand *> outputs;
    SmallVector<Operation *> outputOpChain;
    if (!xsmm::utils::WithOutput(transposeOp,
                                 xsmm::FuncType<vector::TransferWriteOp>,
                                 outputs, outputOpChain)) {
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
    operations.push_back(xsmm::FuncType<vector::ContractionOp>);
    SmallVector<Operation *> opChain;
    if (!xsmm::utils::WithOps(&contractOp->getParentOp()->getRegion(0),
                              contractOp->getParentOp(), operations, opChain)) {
      return failure();
    }
    assert(opChain[0] == contractOp);

    SmallVector<std::function<bool(Operation * op)>> inputOperations;
    inputOperations.push_back(xsmm::FuncType<vector::TransferReadOp>);
    inputOperations.push_back(xsmm::FuncType<vector::TransferReadOp>);
    inputOperations.push_back(xsmm::FuncType<vector::TransferReadOp>);
    SmallVector<OpOperand *> inputs;
    SmallVector<Operation *> inputOpChain;
    if (!xsmm::utils::WithInputs(rewriter, contractOp, inputOperations, inputs,
                                 inputOpChain)) {
      return failure();
    }

    if (contractOp.getKind() != CombiningKind::ADD) {
      return failure();
    }
    auto indexingMaps = contractOp.getIndexingMaps();
    if (indexingMaps.size() != 3) {
      return failure();
    }

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
    operations.push_back(xsmm::FuncType<arith::AddFOp>);
    opChain.clear();
    if (xsmm::utils::WithOps(&contractOp->getParentOp()->getRegion(0),
                             contractOp->getParentOp(), operations, opChain)) {
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
          if (xsmm::utils::WithOps(&contractOp->getParentOp()->getRegion(0),
                                   contractOp->getParentOp(), operations,
                                   opChain)) {
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
              if (xsmm::utils::WithOutput(
                      maxF, xsmm::FuncType<vector::TransferWriteOp>,
                      maxfOutputs, maxfOutputChain)) {
                SmallVector<Operation *> contractOutputChain;
                if (xsmm::utils::WithOutput(
                        contractOp, xsmm::FuncType<vector::TransferWriteOp>,
                        outputs, contractOutputChain)) {

                  auto fusedBrgemmInfo = xsmm::utils::isMappableToBrgemm(
                      rewriter, contractOp, inputs, outputs, affineMaps);
                  if (failed(fusedBrgemmInfo)) {
                    return failure();
                  }
                  vector::TransferWriteOp zeroOp;
                  xsmm::utils::WithZeroInit(inputs[2], zeroOp);
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
      SmallVector<Operation *> contractOutputChain;
      if (xsmm::utils::WithOutput(contractOp,
                                  xsmm::FuncType<vector::TransferWriteOp>,
                                  outputs, contractOutputChain)) {
        auto brgemmInfo = xsmm::utils::isMappableToBrgemm(
            rewriter, contractOp, inputs, outputs, affineMaps);
        if (failed(brgemmInfo)) {
          return failure();
        }
        vector::TransferWriteOp zeroOp;
        xsmm::utils::WithZeroInit(inputs[2], zeroOp);
        replaceOpWithGemmLikeOp(rewriter, contractOp, *brgemmInfo, inputs,
                                zeroOp);
      } else {
        return failure();
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
