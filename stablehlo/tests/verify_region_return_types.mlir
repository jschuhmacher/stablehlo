// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file

// stablehlo.return accepts tokens and tuples; regions requiring tensors must
// diagnose them.

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

func.func @sort_comparator_returns_tuple(%arg0: tensor<3x5xf32>) -> tensor<3x5xf32> {
  // expected-error @+1 {{comparator must return tensor<i1> but got 'tuple<tensor<i1>>'}}
  %0 = "stablehlo.sort"(%arg0) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %less = stablehlo.compare LT, %lhs, %rhs : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %tuple = stablehlo.tuple %less : tuple<tensor<i1>>
    stablehlo.return %tuple : tuple<tensor<i1>>
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

func.func @map_computation_returns_tuple(%arg0: tensor<3x5xf32>) -> tensor<3x5xf32> {
  // expected-error @+2 {{failed to infer returned types}}
  // expected-error @+1 {{computation must return 0-rank tensor, but got: 'tuple<tensor<f32>>'}}
  %0 = "stablehlo.map"(%arg0) ({
  ^bb0(%element: tensor<f32>):
    %tuple = stablehlo.tuple %element : tuple<tensor<f32>>
    stablehlo.return %tuple : tuple<tensor<f32>>
  }) {dimensions = array<i64: 0, 1>} : (tensor<3x5xf32>) -> tensor<3x5xf32>
  func.return %0 : tensor<3x5xf32>
}

// -----

func.func @reduce_body_returns_tuple(%arg0: tensor<3x5xf32>, %init: tensor<f32>) -> tensor<3xf32> {
  // expected-error @+1 {{Reduction-region here must produce tensor-typed result(s), but produces 'tuple<tensor<f32>>' instead}}
  %0 = stablehlo.reduce(%arg0 init: %init) across dimensions = [1] : (tensor<3x5xf32>, tensor<f32>) -> tensor<3xf32>
    reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      %tuple = stablehlo.tuple %sum : tuple<tensor<f32>>
      stablehlo.return %tuple : tuple<tensor<f32>>
    }
  func.return %0 : tensor<3xf32>
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

func.func @scatter_body_returns_tuple(%input: tensor<3x5xf32>, %indices: tensor<2x1xi32>, %updates: tensor<2x5xf32>) -> tensor<3x5xf32> {
  // expected-error @+1 {{Reduction-region here must produce tensor-typed result(s), but produces 'tuple<tensor<f32>>' instead}}
  %0 = "stablehlo.scatter"(%input, %indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %tuple = stablehlo.tuple %rhs : tuple<tensor<f32>>
    stablehlo.return %tuple : tuple<tensor<f32>>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1], inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0], index_vector_dim = 1>
  } : (tensor<3x5xf32>, tensor<2x1xi32>, tensor<2x5xf32>) -> tensor<3x5xf32>
  func.return %0 : tensor<3x5xf32>
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

func.func @if_branch_returns_tuple(%pred: tensor<i1>, %arg0: tensor<3x5xf32>) -> tensor<3x5xf32> {
  // expected-error @+2 {{failed to infer returned types}}
  // expected-error @+1 {{inferred type(s) 'tuple<tensor<3x5xf32>>' are incompatible with return type(s) of operation 'tensor<3x5xf32>'}}
  %0 = "stablehlo.if"(%pred) ({
    %tuple = stablehlo.tuple %arg0 : tuple<tensor<3x5xf32>>
    stablehlo.return %tuple : tuple<tensor<3x5xf32>>
  }, {
    %tuple = stablehlo.tuple %arg0 : tuple<tensor<3x5xf32>>
    stablehlo.return %tuple : tuple<tensor<3x5xf32>>
  }) : (tensor<i1>) -> tensor<3x5xf32>
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

// -----

func.func @while_condition_returns_tuple(%arg0: tensor<3x5xf32>) -> tensor<3x5xf32> {
  // expected-error @+1 {{expect condition block return a zero-ranked tensor of i1 but got 'tuple<tensor<i1>>'}}
  %0 = "stablehlo.while"(%arg0) ({
  ^bb0(%state: tensor<3x5xf32>):
    %continue = stablehlo.constant dense<true> : tensor<i1>
    %tuple = stablehlo.tuple %continue : tuple<tensor<i1>>
    stablehlo.return %tuple : tuple<tensor<i1>>
  }, {
  ^bb0(%state: tensor<3x5xf32>):
    stablehlo.return %state : tensor<3x5xf32>
  }) : (tensor<3x5xf32>) -> tensor<3x5xf32>
  func.return %0 : tensor<3x5xf32>
}
