// NOTE: Assertions have been autogenerated by utils/update_mlir_test_checks.py
// RUN: circt-opt -create-dataflow %s | FileCheck %s
func @affine_apply_ceildiv(%arg0: index) -> index {
// CHECK:       module {

// CHECK-LABEL:   handshake.func @affine_apply_ceildiv(
// CHECK-SAME:                                         %[[VAL_0:.*]]: index, %[[VAL_1:.*]]: none, ...) -> (index, none) {
// CHECK:           %[[VAL_2:.*]] = "handshake.merge"(%[[VAL_0]]) : (index) -> index
// CHECK:           %[[VAL_3:.*]]:3 = "handshake.fork"(%[[VAL_2]]) {control = false} : (index) -> (index, index, index)
// CHECK:           %[[VAL_4:.*]]:4 = "handshake.fork"(%[[VAL_1]]) {control = true} : (none) -> (none, none, none, none)
// CHECK:           %[[VAL_5:.*]] = "handshake.constant"(%[[VAL_4]]#2) {value = 42 : index} : (none) -> index
// CHECK:           %[[VAL_6:.*]] = "handshake.constant"(%[[VAL_4]]#1) {value = 0 : index} : (none) -> index
// CHECK:           %[[VAL_7:.*]]:3 = "handshake.fork"(%[[VAL_6]]) {control = false} : (index) -> (index, index, index)
// CHECK:           %[[VAL_8:.*]] = "handshake.constant"(%[[VAL_4]]#0) {value = 1 : index} : (none) -> index
// CHECK:           %[[VAL_9:.*]]:2 = "handshake.fork"(%[[VAL_8]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_10:.*]] = cmpi "sle", %[[VAL_3]]#2, %[[VAL_7]]#0 : index
// CHECK:           %[[VAL_11:.*]]:2 = "handshake.fork"(%[[VAL_10]]) {control = false} : (i1) -> (i1, i1)
// CHECK:           %[[VAL_12:.*]] = subi %[[VAL_7]]#1, %[[VAL_3]]#1 : index
// CHECK:           %[[VAL_13:.*]] = subi %[[VAL_3]]#0, %[[VAL_9]]#0 : index
// CHECK:           %[[VAL_14:.*]] = select %[[VAL_11]]#1, %[[VAL_12]], %[[VAL_13]] : index
// CHECK:           %[[VAL_15:.*]] = divi_signed %[[VAL_14]], %[[VAL_5]] : index
// CHECK:           %[[VAL_16:.*]]:2 = "handshake.fork"(%[[VAL_15]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_17:.*]] = subi %[[VAL_7]]#2, %[[VAL_16]]#1 : index
// CHECK:           %[[VAL_18:.*]] = addi %[[VAL_16]]#0, %[[VAL_9]]#1 : index
// CHECK:           %[[VAL_19:.*]] = select %[[VAL_11]]#0, %[[VAL_17]], %[[VAL_18]] : index
// CHECK:           handshake.return %[[VAL_19]], %[[VAL_4]]#3 : index, none
// CHECK:         }
// CHECK:       }

    %c42 = constant 42 : index
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %0 = cmpi "sle", %arg0, %c0 : index
    %1 = subi %c0, %arg0 : index
    %2 = subi %arg0, %c1 : index
    %3 = select %0, %1, %2 : index
    %4 = divi_signed %3, %c42 : index
    %5 = subi %c0, %4 : index
    %6 = addi %4, %c1 : index
    %7 = select %0, %5, %6 : index
    return %7 : index
  }
