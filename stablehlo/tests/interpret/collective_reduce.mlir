// RUN: stablehlo-translate --interpret -split-input-file %s

module @cross_replica {
  func.func @collective_reduce(%operand : tensor<4xi64>) -> tensor<4xi64> {
    %result = "stablehlo.collective_reduce"(%operand) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
        stablehlo.return %0 : tensor<i64>
    }) {
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
    } : (tensor<4xi64>) -> tensor<4xi64>
    return %result : tensor<4xi64>
  }
  func.func @main() {
    %inputs0 = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi64>
    %inputs1 = stablehlo.constant dense<[5, 6, 7, 8]> : tensor<4xi64>
    %results:2 = "interpreter.run_parallel"(%inputs0, %inputs1) {
      programs=[[@collective_reduce], [@collective_reduce]]
    } : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>, tensor<4xi64>)
    // Root is process 0 (index 0 in group), result is 1+5=6, 2+6=8, etc.
    check.expect_eq_const %results#0, dense<[6, 8, 10, 12]> : tensor<4xi64>
    // Non-root processes get zeros.
    check.expect_eq_const %results#1, dense<[0, 0, 0, 0]> : tensor<4xi64>
    func.return
  }
}

// -----

module @cross_partition {
  func.func @collective_reduce(%operand : tensor<2xi64>) -> tensor<2xi64> {
    %result = "stablehlo.collective_reduce"(%operand) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
        stablehlo.return %0 : tensor<i64>
    }) {
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>
    } : (tensor<2xi64>) -> tensor<2xi64>
    return %result : tensor<2xi64>
  }
  func.func @main() {
    %r0p0 = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
    %r0p1 = stablehlo.constant dense<[10, 20]> : tensor<2xi64>
    %r1p0 = stablehlo.constant dense<[100, 200]> : tensor<2xi64>
    %r1p1 = stablehlo.constant dense<[1000, 2000]> : tensor<2xi64>
    %results:4 = "interpreter.run_parallel"(%r0p0, %r0p1, %r1p0, %r1p1) {
      programs=[[@collective_reduce, @collective_reduce],
                [@collective_reduce, @collective_reduce]]
    } : (tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) ->
        (tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>)
    // Each replica's partitions form a group rooted at partition 0.
    check.expect_eq_const %results#0, dense<[11, 22]> : tensor<2xi64>
    check.expect_eq_const %results#1, dense<[0, 0]> : tensor<2xi64>
    check.expect_eq_const %results#2, dense<[1100, 2200]> : tensor<2xi64>
    check.expect_eq_const %results#3, dense<[0, 0]> : tensor<2xi64>
    func.return
  }
}

// -----

module @flattened_ids {
  func.func @collective_reduce(%operand : tensor<2xi64>) -> tensor<2xi64> {
    %result = "stablehlo.collective_reduce"(%operand) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
        stablehlo.return %0 : tensor<i64>
    }) {
      replica_groups = dense<[[0, 3], [1, 2]]> : tensor<2x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids
    } : (tensor<2xi64>) -> tensor<2xi64>
    return %result : tensor<2xi64>
  }
  func.func @main() {
    %r0p0 = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
    %r0p1 = stablehlo.constant dense<[10, 20]> : tensor<2xi64>
    %r1p0 = stablehlo.constant dense<[100, 200]> : tensor<2xi64>
    %r1p1 = stablehlo.constant dense<[1000, 2000]> : tensor<2xi64>
    %results:4 = "interpreter.run_parallel"(%r0p0, %r0p1, %r1p0, %r1p1) {
      programs=[[@collective_reduce, @collective_reduce],
                [@collective_reduce, @collective_reduce]]
    } : (tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) ->
        (tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>)
    // Flattened id 0 roots {0, 3}; flattened id 1 roots {1, 2}.
    check.expect_eq_const %results#0, dense<[1001, 2002]> : tensor<2xi64>
    check.expect_eq_const %results#1, dense<[110, 220]> : tensor<2xi64>
    check.expect_eq_const %results#2, dense<[0, 0]> : tensor<2xi64>
    check.expect_eq_const %results#3, dense<[0, 0]> : tensor<2xi64>
    func.return
  }
}
