// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file

// stablehlo.return accepts tokens; regions requiring tensors must diagnose them.

func.func @sort_comparator_returns_token(%arg0: tensor<3x5xf32>) -> tensor<3x5xf32> {
  // expected-error @+1 {{comparator must return tensor<i1> but got '!stablehlo.token'}}
  %0 = "stablehlo.sort"(%arg0) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %token = stablehlo.create_token : !stablehlo.token
    stablehlo.return %token : !stablehlo.token
  }) {dimension = 1 : i64, is_stable = true} : (tensor<3x5xf32>) -> tensor<3x5xf32>
  func.return %0 : tensor<3x5xf32>
}

// -----

func.func @map_computation_returns_token(%arg0: tensor<3x5xf32>) -> tensor<3x5xf32> {
  // expected-error @+2 {{failed to infer returned types}}
  // expected-error @+1 {{computation must return 0-rank tensor, but got: '!stablehlo.token'}}
  %0 = "stablehlo.map"(%arg0) ({
  ^bb0(%element: tensor<f32>):
    %token = stablehlo.create_token : !stablehlo.token
    stablehlo.return %token : !stablehlo.token
  }) {dimensions = array<i64: 0, 1>} : (tensor<3x5xf32>) -> tensor<3x5xf32>
  func.return %0 : tensor<3x5xf32>
}

// -----

func.func @reduce_window_body_returns_token(%arg0: tensor<4x6xf32>, %init: tensor<f32>) -> tensor<2x3xf32> {
  // expected-error @+1 {{Reduction-region here must produce tensor-typed result(s), but produces '!stablehlo.token' instead}}
  %0 = "stablehlo.reduce_window"(%arg0, %init) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %token = stablehlo.create_token : !stablehlo.token
    stablehlo.return %token : !stablehlo.token
  }) {window_dimensions = array<i64: 2, 2>, window_strides = array<i64: 2, 2>} : (tensor<4x6xf32>, tensor<f32>) -> tensor<2x3xf32>
  func.return %0 : tensor<2x3xf32>
}

// -----

func.func @all_reduce_body_returns_token(%arg0: tensor<3x5xf32>) -> tensor<3x5xf32> {
  // expected-error @+1 {{Reduction-region here must produce tensor-typed result(s), but produces '!stablehlo.token' instead}}
  %0 = "stablehlo.all_reduce"(%arg0) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %token = stablehlo.create_token : !stablehlo.token
    stablehlo.return %token : !stablehlo.token
  }) {replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>} : (tensor<3x5xf32>) -> tensor<3x5xf32>
  func.return %0 : tensor<3x5xf32>
}

// -----

func.func @while_condition_returns_token(%arg0: tensor<3x5xf32>) -> tensor<3x5xf32> {
  // expected-error @+1 {{expect condition block return a zero-ranked tensor of i1 but got '!stablehlo.token'}}
  %0 = "stablehlo.while"(%arg0) ({
  ^bb0(%state: tensor<3x5xf32>):
    %token = stablehlo.create_token : !stablehlo.token
    stablehlo.return %token : !stablehlo.token
  }, {
  ^bb0(%state: tensor<3x5xf32>):
    stablehlo.return %state : tensor<3x5xf32>
  }) : (tensor<3x5xf32>) -> tensor<3x5xf32>
  func.return %0 : tensor<3x5xf32>
}
