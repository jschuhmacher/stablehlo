// RUN: stablehlo-opt --stablehlo-aggressive-simplification %s | stablehlo-translate --interpret

// Each zero is folded into its callee, so only the identities may go.
func.func @add_positive_zero(%x: tensor<2x3xf32>) -> tensor<2x3xi32> {
  %zero = stablehlo.constant dense<0.0> : tensor<2x3xf32>
  %0 = stablehlo.add %x, %zero : tensor<2x3xf32>
  %1 = stablehlo.bitcast_convert %0 : (tensor<2x3xf32>) -> tensor<2x3xi32>
  func.return %1 : tensor<2x3xi32>
}

func.func @add_negative_zero(%x: tensor<2x3xf32>) -> tensor<2x3xi32> {
  %zero = stablehlo.constant dense<-0.0> : tensor<2x3xf32>
  %0 = stablehlo.add %x, %zero : tensor<2x3xf32>
  %1 = stablehlo.bitcast_convert %0 : (tensor<2x3xf32>) -> tensor<2x3xi32>
  func.return %1 : tensor<2x3xi32>
}

func.func @subtract_positive_zero(%x: tensor<2x3xf32>) -> tensor<2x3xi32> {
  %zero = stablehlo.constant dense<0.0> : tensor<2x3xf32>
  %0 = stablehlo.subtract %x, %zero : tensor<2x3xf32>
  %1 = stablehlo.bitcast_convert %0 : (tensor<2x3xf32>) -> tensor<2x3xi32>
  func.return %1 : tensor<2x3xi32>
}

func.func @subtract_negative_zero(%x: tensor<2x3xf32>) -> tensor<2x3xi32> {
  %zero = stablehlo.constant dense<-0.0> : tensor<2x3xf32>
  %0 = stablehlo.subtract %x, %zero : tensor<2x3xf32>
  %1 = stablehlo.bitcast_convert %0 : (tensor<2x3xf32>) -> tensor<2x3xi32>
  func.return %1 : tensor<2x3xi32>
}

func.func @main() {
  %x = stablehlo.constant dense<[[-0.0, 0.0, -0.0], [1.5, -0.0, -2.5]]> : tensor<2x3xf32>
  %0 = func.call @add_positive_zero(%x) : (tensor<2x3xf32>) -> tensor<2x3xi32>
  check.expect_eq_const %0, dense<[[0, 0, 0], [1069547520, 0, -1071644672]]> : tensor<2x3xi32>
  %1 = func.call @add_negative_zero(%x) : (tensor<2x3xf32>) -> tensor<2x3xi32>
  check.expect_eq_const %1, dense<[[-2147483648, 0, -2147483648], [1069547520, -2147483648, -1071644672]]> : tensor<2x3xi32>
  %2 = func.call @subtract_positive_zero(%x) : (tensor<2x3xf32>) -> tensor<2x3xi32>
  check.expect_eq_const %2, dense<[[-2147483648, 0, -2147483648], [1069547520, -2147483648, -1071644672]]> : tensor<2x3xi32>
  %3 = func.call @subtract_negative_zero(%x) : (tensor<2x3xf32>) -> tensor<2x3xi32>
  check.expect_eq_const %3, dense<[[0, 0, 0], [1069547520, 0, -1071644672]]> : tensor<2x3xi32>
  func.return
}
