//===- CalyxHelpers.cpp - Calyx helper methods -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Various helper methods for building Calyx programs.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Calyx/CalyxHelpers.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/PatternMatch.h"

namespace circt {
namespace calyx {

calyx::RegisterOp createRegister(Location loc, OpBuilder &builder,
                                 ComponentOp component, size_t width,
                                 Twine prefix) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(component.getBodyBlock());
  return builder.create<RegisterOp>(loc, (prefix + "_reg").str(), width);
}

calyx::RegisterOp createRegister(Location loc, OpBuilder &builder,
                                 ComponentOp component, Type type,
                                 Twine prefix) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(component.getBodyBlock());
  return builder.create<RegisterOp>(loc, (prefix + "_reg").str(), type);
}

hw::ConstantOp createConstant(Location loc, OpBuilder &builder,
                              ComponentOp component, size_t width,
                              size_t value) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(component.getBodyBlock());
  return builder.create<hw::ConstantOp>(loc,
                                        APInt(width, value, /*unsigned=*/true));
}

bool isControlLeafNode(Operation *op) { return isa<calyx::EnableOp>(op); }

DictionaryAttr getMandatoryPortAttr(MLIRContext *ctx, StringRef name) {
  return DictionaryAttr::get(
      ctx, {NamedAttribute(StringAttr::get(ctx, name), UnitAttr::get(ctx))});
}

void addMandatoryComponentPorts(PatternRewriter &rewriter,
                                SmallVectorImpl<calyx::PortInfo> &ports) {
  MLIRContext *ctx = rewriter.getContext();
  ports.push_back({
      rewriter.getStringAttr("clk"),
      rewriter.getI1Type(),
      calyx::Direction::Input,
      getMandatoryPortAttr(ctx, "clk"),
  });
  ports.push_back({
      rewriter.getStringAttr("reset"),
      rewriter.getI1Type(),
      calyx::Direction::Input,
      getMandatoryPortAttr(ctx, "reset"),
  });
  ports.push_back({
      rewriter.getStringAttr("go"),
      rewriter.getI1Type(),
      calyx::Direction::Input,
      getMandatoryPortAttr(ctx, "go"),
  });
  ports.push_back({
      rewriter.getStringAttr("done"),
      rewriter.getI1Type(),
      calyx::Direction::Output,
      getMandatoryPortAttr(ctx, "done"),
  });
}

unsigned handleZeroWidth(int64_t dim) {
  return std::max(llvm::Log2_64_Ceil(dim), 1U);
}

unsigned getTypeSize(Type type) {
  if (auto intType = type.dyn_cast<IntegerType>())
    return intType.getIntOrFloatBitWidth();
  else if (auto hwArrayType = type.dyn_cast<hw::ArrayType>())
    return getTypeSize(hwArrayType.getElementType()) * hwArrayType.getSize();
  else if (auto vectorType = type.dyn_cast<VectorType>())
    return getTypeSize(vectorType.getElementType()) *
           vectorType.getNumElements();
  llvm_unreachable("Unsupported type");
}

hw::ArrayType vectorTypeToHWArrayType(VectorType vectorType) {
  return hw::ArrayType::get(vectorType.getElementType(),
                            vectorType.getNumElements());
}

Type convertType(Type type) {
  if (auto vectorType = type.dyn_cast<VectorType>())
    return vectorTypeToHWArrayType(vectorType);
  return type;
}

} // namespace calyx
} // namespace circt
