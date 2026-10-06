// RUN: split-file %s %t
// RUN: not stablehlo-opt %t/operand.mlir -allow-unregistered-dialect 2>&1 | FileCheck --check-prefix=OPERAND %s
// RUN: not stablehlo-opt %t/output.mlir -allow-unregistered-dialect 2>&1 | FileCheck --check-prefix=OUTPUT %s

//--- operand.mlir
// OPERAND: buffer operand 1[] is used in output_operand_alias more than once
func.func @duplicate_buffer_operand_alias(
    %arg0: tensor<2xf32>, %arg1: memref<2xf32>)
    -> (memref<2xf32>, memref<2xf32>) {
  %0:2 = "stablehlo.custom_call"(%arg0, %arg1) {
    call_target_name = "foo",
    api_version = 4 : i32,
    output_operand_aliases = [
      #stablehlo.output_operand_alias<output_tuple_indices = [0],
        operand_index = 1, operand_tuple_indices = []>,
      #stablehlo.output_operand_alias<output_tuple_indices = [1],
        operand_index = 1, operand_tuple_indices = []>]
  } : (tensor<2xf32>, memref<2xf32>) -> (memref<2xf32>, memref<2xf32>)
  func.return %0#0, %0#1 : memref<2xf32>, memref<2xf32>
}

//--- output.mlir
// OUTPUT: buffer output [0, 0] is used in output_operand_alias more than once
func.func @duplicate_buffer_output_alias(
    %arg0: tuple<tuple<memref<2xf32>>, memref<2xf32>>)
    -> tuple<tuple<memref<2xf32>>> {
  %0 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "foo",
    api_version = 4 : i32,
    output_operand_aliases = [
      #stablehlo.output_operand_alias<output_tuple_indices = [0, 0],
        operand_index = 0, operand_tuple_indices = [0, 0]>,
      #stablehlo.output_operand_alias<output_tuple_indices = [0, 0],
        operand_index = 0, operand_tuple_indices = [1]>]
  } : (tuple<tuple<memref<2xf32>>, memref<2xf32>>) ->
      tuple<tuple<memref<2xf32>>>
  func.return %0 : tuple<tuple<memref<2xf32>>>
}
