//===- CalyxToHW.cpp - Translate Calyx into HW ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This is the main Calyx to HW Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/CalyxToHW.h"
#include "../PassDetail.h"
#include "circt/Dialect/Calyx/CalyxHelpers.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace circt::calyx;
using namespace circt::comb;
using namespace circt::hw;
using namespace circt::seq;
using namespace circt::sv;

/// ConversionPatterns.

struct ConvertComponentOp : public OpConversionPattern<ComponentOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ComponentOp component, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<hw::PortInfo> hwInputInfo;
    auto portInfo = component.getPortInfo();
    for (auto [name, type, direction, _] : portInfo)
      hwInputInfo.push_back({name, hwDirection(direction), type});
    ModulePortInfo hwPortInfo(hwInputInfo);

    SmallVector<Value> argValues;
    auto hwMod = rewriter.create<HWModuleOp>(
        component.getLoc(), component.getNameAttr(), hwPortInfo,
        [&](OpBuilder &b, HWModulePortAccessor &ports) {
          for (auto [name, type, direction, _] : portInfo) {
            switch (direction) {
            case calyx::Direction::Input:
              assert(ports.getInput(name).getType() == type);
              argValues.push_back(ports.getInput(name));
              break;
            case calyx::Direction::Output:
              auto wire = b.create<sv::WireOp>(component.getLoc(), type, name);
              auto wireRead =
                  b.create<sv::ReadInOutOp>(component.getLoc(), wire);
              argValues.push_back(wireRead);
              ports.setOutput(name, wireRead);
              break;
            }
          }
        });

    auto *outputOp = hwMod.getBodyBlock()->getTerminator();
    rewriter.mergeBlocks(component.getBodyBlock(), hwMod.getBodyBlock(),
                         argValues);
    outputOp->moveAfter(&hwMod.getBodyBlock()->back());
    rewriter.eraseOp(component);
    return success();
  }

private:
  hw::PortDirection hwDirection(calyx::Direction dir) const {
    switch (dir) {
    case calyx::Direction::Input:
      return hw::PortDirection::INPUT;
    case calyx::Direction::Output:
      return hw::PortDirection::OUTPUT;
    }
    llvm_unreachable("unknown direction");
  }
};

struct ConvertWiresOp : public OpConversionPattern<WiresOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WiresOp wires, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    HWModuleOp hwMod = wires->getParentOfType<HWModuleOp>();
    rewriter.inlineRegionBefore(wires.getBody(), hwMod.getBodyRegion(),
                                hwMod.getBodyRegion().end());
    rewriter.eraseOp(wires);
    rewriter.inlineBlockBefore(&hwMod.getBodyRegion().getBlocks().back(),
                               &hwMod.getBodyBlock()->back());
    return success();
  }
};

struct ConvertControlOp : public OpConversionPattern<ControlOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ControlOp control, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!control.getBodyBlock()->empty())
      return control.emitOpError("calyx control must be structural");
    rewriter.eraseOp(control);
    return success();
  }
};

struct ConvertAssignOp : public OpConversionPattern<calyx::AssignOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(calyx::AssignOp assign, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = adaptor.getSrc();

    // llvm::errs() << "calyx::AssignOp\n";
    // assign.dump();

    Type srcType = convertType(src.getType());

    if (auto guard = adaptor.getGuard()) {
      Value zero;
      if (auto arrayType = dyn_cast<hw::ArrayType>(srcType)) {
        SmallVector<Value> zeroValues;
        for (size_t i = 0; i < arrayType.getSize(); i++)
          zeroValues.push_back(rewriter.create<hw::ConstantOp>(
              assign.getLoc(), arrayType.getElementType(), 0));
        zero = rewriter.create<hw::ArrayCreateOp>(assign.getLoc(), zeroValues);
      } else {
        zero = rewriter.create<hw::ConstantOp>(assign.getLoc(), srcType, 0);
      }

      src = rewriter.create<MuxOp>(assign.getLoc(), guard, src, zero);
      for (Operation *destUser :
           llvm::make_early_inc_range(assign.getDest().getUsers())) {
        if (destUser == assign)
          continue;
        if (auto otherAssign = dyn_cast<calyx::AssignOp>(destUser)) {
          src = rewriter.create<MuxOp>(assign.getLoc(), otherAssign.getGuard(),
                                       otherAssign.getSrc(), src);
          rewriter.eraseOp(destUser);
        }
      }
    }

    // To make life easy in ConvertComponentOp, we read from the output wires so
    // the dialect conversion block argument mapping would work without a type
    // converter. This means assigns to ComponentOp outputs will try to assign
    // to a read from a wire, so we need to map to the wire.
    Value dest = adaptor.getDest();

    if (auto readInOut = dyn_cast<ReadInOutOp>(dest.getDefiningOp()))
      dest = readInOut.getInput();

    Type dstRawType = dest.getType();
    if (auto inoutType = dstRawType.dyn_cast<circt::hw::InOutType>())
      dstRawType = inoutType.getElementType();
    dstRawType = convertType(dstRawType);

    // llvm::errs() << "srcType: " << srcType << ", dstRawType: " << dstRawType
    //              << "\n";

    if (srcType != dstRawType) {
      // llvm::errs() << "srcType != dstRawType\n";

      int srcTypeSize = getTypeSize(srcType);
      int dstRawTypeSize = getTypeSize(dstRawType);

      if (srcTypeSize > dstRawTypeSize) {
        auto base = rewriter.create<hw::ConstantOp>(
            assign.getLoc(), rewriter.getI32IntegerAttr(0));
        src = rewriter.create<sv::IndexedPartSelectOp>(assign.getLoc(), src,
                                                       base, dstRawTypeSize);
      }

      // array -> bits
      if (hw::ArrayType srcArrayType = srcType.dyn_cast<hw::ArrayType>()) {
        uint32_t srcArrayElementSize =
            srcArrayType.getElementType().getIntOrFloatBitWidth();

        // llvm::errs() << "srcArrayElementSize: " << srcArrayElementSize <<
        // "\n"; llvm::errs() << "srcArray.getSize(): " <<
        // srcArrayType.getSize()
        //              << "\n";

        uint32_t indexBits = llvm::Log2_32_Ceil(srcArrayType.getSize());
        for (size_t i = 0; i < srcArrayType.getSize(); i++) {
          auto base = rewriter.create<hw::ConstantOp>(
              assign.getLoc(),
              rewriter.getI32IntegerAttr(i * srcArrayElementSize));
          auto destIndexedPartSelect =
              rewriter.create<sv::IndexedPartSelectInOutOp>(
                  assign.getLoc(), dest, base, srcArrayElementSize);

          auto srcElementIndex = rewriter.create<hw::ConstantOp>(
              assign.getLoc(),
              rewriter.getIntegerAttr(rewriter.getIntegerType(indexBits), i));
          auto srcElement = rewriter.create<hw::ArrayGetOp>(
              assign.getLoc(), src, srcElementIndex);

          rewriter.create<sv::AssignOp>(assign.getLoc(), destIndexedPartSelect,
                                        srcElement);
        }
      } else if (hw::ArrayType dstArrayType =
                     dstRawType.dyn_cast<hw::ArrayType>()) { // bits -> array
        uint32_t dstArrayElementSize =
            dstArrayType.getElementType().getIntOrFloatBitWidth();
        // llvm::errs() << "dstArrayElementSize: " << dstArrayElementSize <<
        // "\n"; llvm::errs() << "dstArray.getSize(): " <<
        // dstArrayType.getSize()
        //              << "\n";

        uint32_t indexBits = llvm::Log2_32_Ceil(dstArrayType.getSize());
        for (size_t i = 0; i < dstArrayType.getSize(); i++) {
          auto base = rewriter.create<hw::ConstantOp>(
              assign.getLoc(),
              rewriter.getI32IntegerAttr(i * dstArrayElementSize));
          auto srcIndexedPartSelect = rewriter.create<sv::IndexedPartSelectOp>(
              assign.getLoc(), src, base, dstArrayElementSize);

          auto dstElementIndex = rewriter.create<hw::ConstantOp>(
              assign.getLoc(),
              rewriter.getIntegerAttr(rewriter.getIntegerType(indexBits), i));
          auto dstElement = rewriter.create<sv::ArrayIndexInOutOp>(
              assign.getLoc(), dest, dstElementIndex);

          rewriter.create<sv::AssignOp>(assign.getLoc(), dstElement,
                                        srcIndexedPartSelect);
        }
      } else {
        rewriter.create<sv::AssignOp>(assign.getLoc(), dest, src);
      }
      rewriter.eraseOp(assign);
    } else {
      rewriter.replaceOpWithNewOp<sv::AssignOp>(assign, dest, src);
    }

    return success();
  }
};

struct ConvertArithConstantOp : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::errs() << "arith::ConstantOp\n";
    op.dump();

    auto vectorType = op.getType().dyn_cast<VectorType>();
    assert(vectorType);

    SmallVector<Value> values;
    for (auto value :
         op.getValueAttr().cast<DenseElementsAttr>().getValues<IntegerAttr>()) {
      auto valueConst =
          rewriter.create<hw::ConstantOp>(op.getLoc(), value.getValue());
      values.push_back(valueConst);
    }

    rewriter.replaceOpWithNewOp<hw::ArrayCreateOp>(op, convertType(vectorType),
                                                   values);
    return success();
  }
};

struct ConvertCellOp : public OpInterfaceConversionPattern<CellInterface> {
  using OpInterfaceConversionPattern::OpInterfaceConversionPattern;

  LogicalResult
  matchAndRewrite(CellInterface cell, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    assert(operands.empty() && "calyx cells do not have operands");

    SmallVector<Value> wires;
    ImplicitLocOpBuilder builder(cell.getLoc(), rewriter);
    convertPrimitiveOp(cell, wires, builder);
    if (wires.size() != cell.getPortInfo().size()) {
      auto diag = cell.emitOpError("couldn't convert to core primitive");
      llvm::errs() << "wires.size(): " << wires.size() << " vs "
                   << "cell.getPortInfo().size(): " << cell.getPortInfo().size()
                   << "\n";
      for (Value wire : wires)
        diag.attachNote() << "with wire: " << wire;
      return diag;
    }

    rewriter.replaceOp(cell, wires);

    return success();
  }

private:
  void convertPrimitiveOp(Operation *op, SmallVectorImpl<Value> &wires,
                          ImplicitLocOpBuilder &b) const {
    TypeSwitch<Operation *>(op)
        // Comparison operations.
        .Case([&](EqLibOp op) {
          convertCompareBinaryOp(op, ICmpPredicate::eq, wires, b);
        })
        .Case([&](NeqLibOp op) {
          convertCompareBinaryOp(op, ICmpPredicate::ne, wires, b);
        })
        .Case([&](LtLibOp op) {
          convertCompareBinaryOp(op, ICmpPredicate::ult, wires, b);
        })
        .Case([&](LeLibOp op) {
          convertCompareBinaryOp(op, ICmpPredicate::ule, wires, b);
        })
        .Case([&](GtLibOp op) {
          convertCompareBinaryOp(op, ICmpPredicate::ugt, wires, b);
        })
        .Case([&](GeLibOp op) {
          convertCompareBinaryOp(op, ICmpPredicate::uge, wires, b);
        })
        .Case([&](SltLibOp op) {
          convertCompareBinaryOp(op, ICmpPredicate::slt, wires, b);
        })
        .Case([&](SleLibOp op) {
          convertCompareBinaryOp(op, ICmpPredicate::sle, wires, b);
        })
        .Case([&](SgtLibOp op) {
          convertCompareBinaryOp(op, ICmpPredicate::sgt, wires, b);
        })
        .Case([&](SgeLibOp op) {
          convertCompareBinaryOp(op, ICmpPredicate::sge, wires, b);
        })
        // Combinational arithmetic and logical operations.
        .Case([&](AddLibOp op) {
          convertArithBinaryOp<AddLibOp, AddOp>(op, wires, b);
        })
        .Case([&](SubLibOp op) {
          convertArithBinaryOp<SubLibOp, SubOp>(op, wires, b);
        })
        .Case([&](RshLibOp op) {
          convertArithBinaryOp<RshLibOp, ShrUOp>(op, wires, b);
        })
        .Case([&](SrshLibOp op) {
          convertArithBinaryOp<SrshLibOp, ShrSOp>(op, wires, b);
        })
        .Case([&](LshLibOp op) {
          convertArithBinaryOp<LshLibOp, ShlOp>(op, wires, b);
        })
        .Case([&](AndLibOp op) {
          convertArithBinaryOp<AndLibOp, AndOp>(op, wires, b);
        })
        .Case([&](OrLibOp op) {
          convertArithBinaryOp<OrLibOp, OrOp>(op, wires, b);
        })
        .Case([&](XorLibOp op) {
          convertArithBinaryOp<XorLibOp, XorOp>(op, wires, b);
        })
        // Pipelined arithmetic operations.
        .Case([&](MultPipeLibOp op) {
          convertPipelineOp<MultPipeLibOp, comb::MulOp>(op, wires, b);
        })
        .Case([&](DivUPipeLibOp op) {
          convertPipelineOp<DivUPipeLibOp, comb::DivUOp>(op, wires, b);
        })
        .Case([&](DivSPipeLibOp op) {
          convertPipelineOp<DivSPipeLibOp, comb::DivSOp>(op, wires, b);
        })
        .Case([&](RemSPipeLibOp op) {
          convertPipelineOp<RemSPipeLibOp, comb::ModSOp>(op, wires, b);
        })
        .Case([&](RemUPipeLibOp op) {
          convertPipelineOp<RemUPipeLibOp, comb::ModUOp>(op, wires, b);
        })
        // Sequential operations.
        .Case([&](RegisterOp op) {
          auto in =
              wireIn(op.getIn(), op.instanceName(), op.portName(op.getIn()), b);
          auto writeEn = wireIn(op.getWriteEn(), op.instanceName(),
                                op.portName(op.getWriteEn()), b);
          auto clk = wireIn(op.getClk(), op.instanceName(),
                            op.portName(op.getClk()), b);
          auto reset = wireIn(op.getReset(), op.instanceName(),
                              op.portName(op.getReset()), b);
          auto doneReg =
              reg(writeEn, clk, reset, op.instanceName() + "_done_reg", b);
          auto done =
              wireOut(doneReg, op.instanceName(), op.portName(op.getDone()), b);
          auto clockEn = b.create<AndOp>(writeEn, createOrFoldNot(done, b));
          auto outReg =
              regCe(in, clk, clockEn, reset, op.instanceName() + "_reg", b);
          auto out = wireOut(outReg, op.instanceName(), "", b);
          wires.append({in.getInput(), writeEn.getInput(), clk.getInput(),
                        reset.getInput(), out, done});
        })
        .Case([&](MemoryOp op) {
          auto clk =
              wireIn(op.clk(), op.instanceName(), op.portName(op.clk()), b);
          auto reset =
              wireIn(op.reset(), op.instanceName(), op.portName(op.reset()), b);

          SmallVector<Value> addrPorts /*, addrPortsShifted*/;
          for (const auto &port : op.addrPorts()) {
            llvm::errs() << "port: " << port << "\n";
            auto addr = wireIn(port, op.instanceName(), op.portName(port), b);
            addrPorts.push_back(addr);
#if 0
            // Addrs address individual bytes.
            int shiftBits = llvm::Log2_32_Ceil(op.getWidth() / 8);
            llvm::errs() << "op.getWidth() " << op.getWidth() << "\n";
            llvm::errs() << "shiftBits " << shiftBits << "\n";
            auto shiftBitsCst = b.create<hw::ConstantOp>(
                b.getIntegerAttr(port.getType(), shiftBits));
            auto shifted = b.create<comb::ShrUOp>(addr, shiftBitsCst);
            addrPortsShifted.push_back(shifted);
#endif
#if 0
            // HACK: Get rid of offset bits
            // llvm::errs() << "Has attr? " << op->hasAttr("calyx.lanes") <<
            // "\n";
            if (op->hasAttr("calyx.lanes")) {
              int lanes =
                  op->getAttr("calyx.lanes").cast<IntegerAttr>().getInt();
              int offsetBits = llvm::Log2_32_Ceil(lanes);
              llvm::errs() << "calyx.lanes " << lanes << "\n";
              llvm::errs() << "offsetBits " << offsetBits << "\n";
              auto offsetBitsCst = b.create<hw::ConstantOp>(
                  b.getIntegerAttr(port.getType(), offsetBits));
              auto shifted = b.create<comb::ShrUOp>(addr, offsetBitsCst);
              addrPortsShifted.push_back(shifted);

            }
#endif
          }

          auto writeData = wireIn(op.writeData(), op.instanceName(),
                                  op.portName(op.writeData()), b);
          auto writeEn = wireIn(op.writeEn(), op.instanceName(),
                                op.portName(op.writeEn()), b);
          auto readEn = wireIn(op.readEn(), op.instanceName(),
                               op.portName(op.readEn()), b);

          auto en = b.create<OrOp>(writeEn, readEn);

          auto doneReg =
              reg(en, clk, reset, op.instanceName() + "_done_reg", b);
          auto done = wireOut(doneReg, op.instanceName(), "", b);

          auto writeEnMsk = b.create<AndOp>(writeEn, createOrFoldNot(done, b));

          SmallVector<int64_t> shape;
          for (auto size : op.getSizes())
            shape.push_back(size.dyn_cast<IntegerAttr>().getInt());

          seq::HLMemOp hlMem = b.create<seq::HLMemOp>(
              clk, reset, b.getStringAttr(op.instanceName() + "_mem"), shape,
              b.getIntegerType(op.getWidth()));

          seq::WritePortOp writePort = b.create<seq::WritePortOp>(
              hlMem, addrPorts, writeData, writeEnMsk,
              /*latency=*/1);
          seq::ReadPortOp readPort = b.create<seq::ReadPortOp>(
              hlMem, addrPorts, readEn, /*latency=*/0);

          auto readData = wireOut(readPort, op.instanceName(), "", b);

          wires.append(addrPorts);
          wires.append({writeData.getInput(), writeEn.getInput(),
                        clk.getInput(), reset.getInput(), readData,
                        readEn.getInput(), done});
        })
        // Unary operqations.
        .Case([&](SliceLibOp op) {
          auto in =
              wireIn(op.getIn(), op.instanceName(), op.portName(op.getIn()), b);
          auto outWidth = op.getOut().getType().getIntOrFloatBitWidth();

          auto extract = b.create<ExtractOp>(in, 0, outWidth);

          auto out =
              wireOut(extract, op.instanceName(), op.portName(op.getOut()), b);
          wires.append({in.getInput(), out});
        })
        .Case([&](NotLibOp op) {
          auto in =
              wireIn(op.getIn(), op.instanceName(), op.portName(op.getIn()), b);

          auto notOp = comb::createOrFoldNot(in, b);

          auto out =
              wireOut(notOp, op.instanceName(), op.portName(op.getOut()), b);
          wires.append({in.getInput(), out});
        })
        .Case([&](WireLibOp op) {
          auto wire = wireIn(op.getIn(), op.instanceName(), "", b);
          wires.append({wire.getInput(), wire});
        })
        .Case([&](PadLibOp op) {
          auto in =
              wireIn(op.getIn(), op.instanceName(), op.portName(op.getIn()), b);
          auto srcWidth = in.getType().getIntOrFloatBitWidth();
          auto destWidth = op.getOut().getType().getIntOrFloatBitWidth();
          auto zero = b.create<hw::ConstantOp>(op.getLoc(),
                                               APInt(destWidth - srcWidth, 0));
          auto padded = wireOut(b.createOrFold<comb::ConcatOp>(zero, in),
                                op.instanceName(), op.portName(op.getOut()), b);
          wires.append({in.getInput(), padded});
        })
        .Case([&](ExtSILibOp op) {
          auto in =
              wireIn(op.getIn(), op.instanceName(), op.portName(op.getIn()), b);
          auto extsi = wireOut(
              createOrFoldSExt(op.getLoc(), in, op.getOut().getType(), b),
              op.instanceName(), op.portName(op.getOut()), b);
          wires.append({in.getInput(), extsi});
        })
        .Case([&](SplatLibOp op) {
          auto in =
              wireIn(op.getIn(), op.instanceName(), op.portName(op.getIn()), b);

          VectorType outType = op.getOut().getType().dyn_cast<VectorType>();
          assert(outType);
          SmallVector<Value> values;
          for (int64_t i = 0; i < outType.getNumElements(); i++)
            values.push_back(in);
          auto array = b.create<hw::ArrayCreateOp>(
              vectorTypeToHWArrayType(outType), values);

          auto out =
              wireOut(array, op.instanceName(), op.portName(op.getOut()), b);
          wires.append({in.getInput(), out});
        })
        .Default([](Operation *) { return SmallVector<Value>(); });
  }

  template <typename OpTy, typename ResultTy>
  void convertArithBinaryOp(OpTy op, SmallVectorImpl<Value> &wires,
                            ImplicitLocOpBuilder &b) const {
    auto left =
        wireIn(op.getLeft(), op.instanceName(), op.portName(op.getLeft()), b);
    auto right =
        wireIn(op.getRight(), op.instanceName(), op.portName(op.getRight()), b);

    Value add;
    if (auto arrayType = dyn_cast<hw::ArrayType>(left.getType())) {
      SmallVector<Value> lanes(arrayType.getSize());
      for (size_t i = 0; i < arrayType.getSize(); i++) {
        auto index = b.create<hw::ConstantOp>(b.getIntegerAttr(
            b.getIntegerType(llvm::Log2_32_Ceil(arrayType.getSize())), i));
        auto l = b.createOrFold<hw::ArrayGetOp>(left, index);
        auto r = b.createOrFold<hw::ArrayGetOp>(right, index);
        lanes[lanes.size() - i - 1] = b.create<ResultTy>(l, r, false);
      }
      add = b.create<hw::ArrayCreateOp>(lanes);
    } else {
      add = b.create<ResultTy>(left, right, false);
    }

    auto out = wireOut(add, op.instanceName(), op.portName(op.getOut()), b);
    wires.append({left.getInput(), right.getInput(), out});
  }

  template <typename OpTy>
  void convertCompareBinaryOp(OpTy op, ICmpPredicate pred,
                              SmallVectorImpl<Value> &wires,
                              ImplicitLocOpBuilder &b) const {
    auto left =
        wireIn(op.getLeft(), op.instanceName(), op.portName(op.getLeft()), b);
    auto right =
        wireIn(op.getRight(), op.instanceName(), op.portName(op.getRight()), b);

    auto add = b.create<ICmpOp>(pred, left, right, false);

    auto out = wireOut(add, op.instanceName(), op.portName(op.getOut()), b);
    wires.append({left.getInput(), right.getInput(), out});
  }

  template <typename SrcOpTy, typename TargetOpTy>
  void convertPipelineOp(SrcOpTy op, SmallVectorImpl<Value> &wires,
                         ImplicitLocOpBuilder &b) const {
    auto clk =
        wireIn(op.getClk(), op.instanceName(), op.portName(op.getClk()), b);
    auto reset =
        wireIn(op.getReset(), op.instanceName(), op.portName(op.getReset()), b);
    auto go = wireIn(op.getGo(), op.instanceName(), op.portName(op.getGo()), b);
    auto left =
        wireIn(op.getLeft(), op.instanceName(), op.portName(op.getLeft()), b);
    auto right =
        wireIn(op.getRight(), op.instanceName(), op.portName(op.getRight()), b);
    wires.append({clk.getInput(), reset.getInput(), go.getInput(),
                  left.getInput(), right.getInput()});

    auto doneReg = reg(go, clk, reset,
                       op.instanceName() + "_" + op.portName(op.getDone()), b);
    auto done =
        wireOut(doneReg, op.instanceName(), op.portName(op.getDone()), b);
    auto clockEn = b.create<AndOp>(go, createOrFoldNot(done, b));

    if (auto arrayType = dyn_cast<hw::ArrayType>(left.getType())) {
      SmallVector<TargetOpTy> lanes(arrayType.getSize());
      for (size_t i = 0; i < arrayType.getSize(); i++) {
        auto index = b.create<hw::ConstantOp>(b.getIntegerAttr(
            b.getIntegerType(llvm::Log2_32_Ceil(arrayType.getSize())), i));
        auto l = b.createOrFold<hw::ArrayGetOp>(left, index);
        auto r = b.createOrFold<hw::ArrayGetOp>(right, index);
        TargetOpTy val = b.create<TargetOpTy>(l, r, false);
        lanes[lanes.size() - i - 1] = val;
      }
      for (size_t i = 0; i < llvm::range_size(op.getOutputPorts()); i++) {
        SmallVector<Value> resultValues;
        for (auto &lane : lanes) {
          resultValues.push_back(lane->getResult(i));
        }
        auto result = b.create<hw::ArrayCreateOp>(arrayType, resultValues);
        auto portName = op.portName(op.getOutputPorts()[i]);
        auto resReg = regCe(result, clk, clockEn, reset,
                            createName(op.instanceName(), portName), b);
        wires.push_back(wireOut(resReg, op.instanceName(), portName, b));
        break;
      }
    } else {
      auto targetOp = b.create<TargetOpTy>(left, right, false);
      for (auto &&[targetRes, sourceRes] :
           llvm::zip(targetOp->getResults(), op.getOutputPorts())) {
        auto portName = op.portName(sourceRes);
        auto resReg = regCe(targetRes, clk, clockEn, reset,
                            createName(op.instanceName(), portName), b);
        wires.push_back(wireOut(resReg, op.instanceName(), portName, b));
      }
    }

    wires.push_back(done);
  }

  ReadInOutOp wireIn(Value source, StringRef instanceName, StringRef portName,
                     ImplicitLocOpBuilder &b) const {
    auto wire = b.create<sv::WireOp>(calyx::convertType(source.getType()),
                                     createName(instanceName, portName));
    return b.create<ReadInOutOp>(wire);
  }

  ReadInOutOp wireOut(Value source, StringRef instanceName, StringRef portName,
                      ImplicitLocOpBuilder &b) const {
    auto wire = b.create<sv::WireOp>(calyx::convertType(source.getType()),
                                     createName(instanceName, portName));
    b.create<sv::AssignOp>(wire, source);
    return b.create<ReadInOutOp>(wire);
  }

  CompRegOp reg(Value source, Value clock, Value reset, Twine name,
                ImplicitLocOpBuilder &b) const {
    auto resetValue = b.create<hw::ConstantOp>(source.getType(), 0);
    auto regName = b.getStringAttr(name);
    return b.create<CompRegOp>(source.getType(), source, clock, regName, reset,
                               resetValue, regName);
  }

  CompRegClockEnabledOp regCe(Value source, Value clock, Value ce, Value reset,
                              Twine name, ImplicitLocOpBuilder &b) const {
    Value resetValue;
    if (auto arrayType = dyn_cast<hw::ArrayType>(source.getType())) {
      SmallVector<Value> resetValues;
      for (size_t i = 0; i < arrayType.getSize(); i++)
        resetValues.push_back(
            b.create<hw::ConstantOp>(arrayType.getElementType(), 0));
      resetValue = b.create<hw::ArrayCreateOp>(resetValues);
    } else {
      resetValue = b.create<hw::ConstantOp>(source.getType(), 0);
    }
    auto regName = b.getStringAttr(name);
    return b.create<CompRegClockEnabledOp>(source.getType(), source, clock, ce,
                                           regName, reset, resetValue, regName);
  }

  std::string createName(StringRef instanceName, StringRef portName) const {
    std::string name = instanceName.str();
    if (!portName.empty())
      name += ("_" + portName).str();
    return name;
  }
};

/// Pass entrypoint.

namespace {
class CalyxToHWPass : public CalyxToHWBase<CalyxToHWPass> {
public:
  void runOnOperation() override;

private:
  LogicalResult runOnModule(ModuleOp module);
};
} // end anonymous namespace

void CalyxToHWPass::runOnOperation() {
  ModuleOp mod = getOperation();
  if (failed(runOnModule(mod)))
    return signalPassFailure();
}

LogicalResult CalyxToHWPass::runOnModule(ModuleOp module) {
  MLIRContext &context = getContext();

  ConversionTarget target(context);
  target.addIllegalDialect<CalyxDialect>();
  target.addLegalDialect<HWDialect>();
  target.addLegalDialect<CombDialect>();
  target.addLegalDialect<SeqDialect>();
  target.addLegalDialect<SVDialect>();

  RewritePatternSet patterns(&context);
  patterns.add<ConvertComponentOp>(&context);
  patterns.add<ConvertWiresOp>(&context);
  patterns.add<ConvertControlOp>(&context);
  patterns.add<ConvertCellOp>(&context);
  patterns.add<ConvertAssignOp>(&context);
  patterns.add<ConvertArithConstantOp>(&context);

  return applyPartialConversion(module, target, std::move(patterns));
}

std::unique_ptr<mlir::Pass> circt::createCalyxToHWPass() {
  return std::make_unique<CalyxToHWPass>();
}
