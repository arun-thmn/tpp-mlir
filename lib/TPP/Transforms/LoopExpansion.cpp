//===- LoopExpansion.cpp -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file  splits parallel loop into scf fors.
//
//===----------------------------------------------------------------------===//
#include "TPP/Dialect/Xsmm/XsmmOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "loop-expansion"

namespace mlir {
namespace tpp {
#define GEN_PASS_DECL_LOOPEXPANSIONPASS
#define GEN_PASS_DEF_LOOPEXPANSIONPASS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;
using namespace mlir::scf;
using namespace std;

namespace mlir {
namespace tpp {

static LogicalResult loopExpand(scf::ForallOp op, unsigned numOuterParallel) {
  OpBuilder b(op);
  IRRewriter rewriter(b.getContext());
  if (numOuterParallel > op.getInductionVars().size()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Number of parallel levels exceeds number of levels of loop");
    return failure();
  }
  if (op.getStaticLowerBound().empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Loop does not have static bounds");
    return failure();
  }
  auto ub = op.getStaticUpperBound().begin();
  auto lb = op.getStaticLowerBound().begin();
  auto step = op.getStaticStep().begin();
  rewriter.setInsertionPointAfter(op);

  SmallVector<scf::ParallelOp> parallelOpList;
  SmallVector<scf::ForOp> forOpList;
  size_t i = 0;
  for (;
       lb != op.getStaticLowerBound().end() &&
       ub != op.getStaticUpperBound().end() && step != op.getStaticStep().end();
       lb++, ub++, step++) {
    auto lowerBound = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), *lb);
    auto upperBound = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), *ub);
    auto stepVal = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), *step);

    if (i < numOuterParallel) {
      auto parallelOp = rewriter.create<scf::ParallelOp>(
          op.getLoc(), ValueRange{lowerBound}, ValueRange{upperBound},
          ValueRange{stepVal});
      rewriter.setInsertionPoint(&parallelOp.getBody()->front());
      parallelOpList.push_back(parallelOp);
      i++;
    } else {
      auto forOp = rewriter.create<scf::ForOp>(op.getLoc(), lowerBound,
                                               upperBound, stepVal);
      // Change the insertion point to the created for loop
      rewriter.setInsertionPoint(&forOp.getBody()->front());
      forOpList.push_back(forOp);
    }
  }

  for (auto oper = op.getBody()->getOperations().begin();
       oper != op.getBody()->getOperations().end(); oper++) {
    if (isa<scf::InParallelOp>(oper)) {
      auto nestedOperations = dyn_cast<scf::InParallelOp>(oper);
      for (auto nestedOper = nestedOperations.begin();
           nestedOper != nestedOperations.end(); nestedOper++) {
        if (isa<scf::InParallelOp>(nestedOper)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Serialization of nested parallel ops unsupported");
          return failure();
        }
      }
    }
  }
  // Clone instructions at the innermost loop level
  IRMapping mapping;
  std::map<Operation *, Operation *> clonedInstrMap;
  for (auto oper = op.getBody()->getOperations().begin();
       oper != op.getBody()->getOperations().end(); oper++) {
    if (!isa<scf::InParallelOp>(oper)) {
      auto clonedInstr = rewriter.clone(*oper, mapping);
      clonedInstrMap[&*oper] = clonedInstr;
    } else {
      auto nestedOperations = (dyn_cast<scf::InParallelOp>(oper));
      for (auto nestedOper = nestedOperations.begin();
           nestedOper != nestedOperations.end(); nestedOper++) {
        auto clonedInstr = rewriter.clone(*nestedOper, mapping);
        clonedInstrMap[&*nestedOper] = clonedInstr;
      }
    }
  }
  for (auto oper = op.getBody()->getOperations().begin();
       oper != op.getBody()->getOperations().end(); oper++) {
    if (!isa<scf::InParallelOp>(oper)) {
      auto clonedInstr = clonedInstrMap[&*oper];
      oper->replaceAllUsesWith(clonedInstr);
      int j = 0;
      for (auto arg : clonedInstr->getOperands()) {
        for (size_t i = 0; i < op.getInductionVars().size(); i++) {
          if (arg == op.getInductionVars()[i]) {
            if (i < numOuterParallel) {
              clonedInstr->setOperand(j,
                                      parallelOpList[i].getInductionVars()[0]);
            } else {
              clonedInstr->setOperand(
                  j, forOpList[i - numOuterParallel].getInductionVar());
            }
            break;
          }
        }
        j++;
      }
    } else {
      auto nestedOperations = (dyn_cast<scf::InParallelOp>(oper));
      for (auto nestedOper = nestedOperations.begin();
           nestedOper != nestedOperations.end(); nestedOper++) {
        auto clonedInstr = clonedInstrMap[&*nestedOper];
        nestedOper->replaceAllUsesWith(clonedInstr);
        int j = 0;
        for (auto arg : clonedInstr->getOperands()) {
          for (size_t i = 0; i < op.getInductionVars().size(); i++) {
            if (arg == op.getInductionVars()[i]) {
              if (i < numOuterParallel) {
                clonedInstr->setOperand(
                    j, parallelOpList[i].getInductionVars()[0]);
              } else {
                clonedInstr->setOperand(
                    j, forOpList[i - numOuterParallel].getInductionVar());
              }
              break;
            }
          }
          j++;
        }
      }
    }
  }

  rewriter.eraseOp(op);
  return success();
}

struct LoopExpansionPass
    : public impl::LoopExpansionPassBase<LoopExpansionPass> {

  using LoopExpansionPassBase::LoopExpansionPassBase;

  void runOnOperation() override {
    getOperation()->walk([&](scf::ForallOp forallOp) {
      if (failed(loopExpand(forallOp, numOuterParallel)))
        LLVM_DEBUG(llvm::dbgs() << "\nFailed to expand the loop\n");
      return WalkResult::advance();
    });
  }
};
} // namespace tpp
} // namespace mlir
