//===- ConvertVectorTypeToHWArrayType.cpp - --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the VectorType to HW ArrayType conversion pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Transforms/Passes.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;

namespace {

static FunctionType convertFunctionType(TypeConverter &typeConverter,
                                        FunctionType type) {
  llvm::SmallVector<Type> res, arg;
  llvm::transform(type.getResults(), std::back_inserter(res),
                  [&](Type t) { return typeConverter.convertType(t); });
  llvm::transform(type.getInputs(), std::back_inserter(arg),
                  [&](Type t) { return typeConverter.convertType(t); });

  return FunctionType::get(type.getContext(), arg, res);
}

class VectorTypeToHWArrayTypeConverter : public mlir::TypeConverter {
public:
  VectorTypeToHWArrayTypeConverter() {
    addConversion([](Type type) -> Type {
      if (auto vectorType = type.dyn_cast<VectorType>()) {
        type.dump();
        return hw::ArrayType::get(vectorType.getElementType(),
                                  vectorType.getNumElements());
      }
      return type;
    });
  }
};

struct TypeConversionPattern : public ConversionPattern {
public:
  TypeConversionPattern(MLIRContext *context, TypeConverter &converter)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, context) {}
  using ConversionPattern::ConversionPattern;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<NamedAttribute, 4> newAttrs;
    newAttrs.reserve(op->getAttrs().size());
    for (auto attr : op->getAttrs()) {
      if (auto typeAttr = attr.getValue().dyn_cast<TypeAttr>()) {
        auto innerType = typeAttr.getValue();
        if (auto funcType = innerType.dyn_cast<FunctionType>(); innerType) {
          innerType = convertFunctionType(*getTypeConverter(), funcType);
        } else {
          innerType = getTypeConverter()->convertType(innerType);
        }
        newAttrs.emplace_back(attr.getName(), TypeAttr::get(innerType));
      } else {
        newAttrs.push_back(attr);
      }
    }

    llvm::SmallVector<Type, 4> newResults;
    (void)getTypeConverter()->convertTypes(op->getResultTypes(), newResults);

    OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                         newResults, newAttrs, op->getSuccessors());

    for (Region &region : op->getRegions()) {
      llvm::SmallVector<Location, 4> argLocs;
      for (auto arg : region.getArguments())
        argLocs.push_back(arg.getLoc());

      Region *newRegion = state.addRegion();
      rewriter.inlineRegionBefore(region, *newRegion, newRegion->begin());
      TypeConverter::SignatureConversion result(newRegion->getNumArguments());
      (void)getTypeConverter()->convertSignatureArgs(
          newRegion->getArgumentTypes(), result);
      rewriter.applySignatureConversion(newRegion, result);

      // Apply the argument locations.
      for (auto [arg, loc] : llvm::zip(newRegion->getArguments(), argLocs))
        arg.setLoc(loc);
    }

    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

static bool isVectorType(Type type) {
  auto match = llvm::TypeSwitch<Type, bool>(type)
                   .Case<VectorType>([](auto type) { return true; })
                   .Case<hw::ArrayType>([](auto type) {
                     return isVectorType(type.getElementType());
                   })
                   .Case<hw::StructType>([](auto type) {
                     return llvm::any_of(type.getElements(), [](auto element) {
                       return isVectorType(element.type);
                     });
                   })
                   .Case<hw::InOutType>([](auto type) {
                     return isVectorType(type.getElementType());
                   })
                   .Default([](auto type) { return false; });

  return match;
}

static bool isVectorAttr(Attribute attr) {
  if (auto typeAttr = attr.dyn_cast<TypeAttr>())
    return isVectorType(typeAttr.getValue());
  return false;
}

static bool isLegalOp(Operation *op) {
  if (auto funcOp = dyn_cast<FunctionOpInterface>(op)) {
    return llvm::none_of(funcOp.getArgumentTypes(), isVectorType) &&
           llvm::none_of(funcOp.getResultTypes(), isVectorType) &&
           llvm::none_of(funcOp.getFunctionBody().getArgumentTypes(),
                         isVectorType);
  }
  auto attrs = llvm::map_range(op->getAttrs(), [](const NamedAttribute &attr) {
    return attr.getValue();
  });

  bool operandsOK = llvm::none_of(op->getOperandTypes(), isVectorType);
  bool resultsOK = llvm::none_of(op->getResultTypes(), isVectorType);
  bool attrsOK = llvm::none_of(attrs, isVectorAttr);
  return operandsOK && resultsOK && attrsOK;
}

struct VectorTypeToHWArrayTypePass
    : public ConvertVectorTypeToHWArrayTypeBase<VectorTypeToHWArrayTypePass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();

    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal(isLegalOp);
    RewritePatternSet patterns(&getContext());
    VectorTypeToHWArrayTypeConverter typeConverter;

    patterns.add<TypeConversionPattern>(patterns.getContext(), typeConverter);

    if (failed(applyFullConversion(module, target, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace circt {
std::unique_ptr<mlir::Pass> createVectorTypeToHWArrayTypeConversion() {
  return std::make_unique<VectorTypeToHWArrayTypePass>();
}

} // namespace circt
