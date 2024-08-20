//RUN: tpp-opt --tile-consumer-and-fuse-producers --bufferize --linalg-generalize-named-ops  --tile-linalg="mTile=4,8 nTile=8,16" ./pass-tile-linalg-chained-gemm.mlir



#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
module {
  func.func @entry(%arg0: tensor<16x32x16x32xf32>, %arg1: tensor<32x32x32x32xf32>, %arg2: tensor<16x32x16x32xf32>) -> tensor<16x32x16x32xf32> {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<16x32x16x32xf32>, tensor<32x32x32x32xf32>) outs(%arg2 : tensor<16x32x16x32xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<16x32x16x32xf32>
    return %0 : tensor<16x32x16x32xf32>
  }
}
[athangam@pcl-tiergarten-login Passes]$ ^C
[athangam@pcl-tiergarten-login Passes]$ vi pass-tile-linalg-gemm.mlir
[athangam@pcl-tiergarten-login Passes]$ cat ../../build/bin/chained_gemm_f32.mlir
// RUN: tpp-run %s -n 10 \
// RUN:  -e entry -entry-point-result=void

// BENCH_TOTAL_FLOPS: 6291456

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
module {
  func.func @entry(%arg0: tensor<64x16x4x4xf32>) -> tensor<64x16x4x4xf32> {
    %cst = arith.constant dense<1.000000e+00> : tensor<16x16x4x4xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<64x16x4x4xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<64x16x4x4xf32>) -> tensor<64x16x4x4xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %cst : tensor<64x16x4x4xf32>, tensor<16x16x4x4xf32>) outs(%1 : tensor<64x16x4x4xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %9 = arith.mulf %in, %in_5 : f32
      %10 = arith.addf %out, %9 : f32
      linalg.yield %10 : f32
    } -> tensor<64x16x4x4xf32>
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<16x16x4x4xf32>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %3 = tensor.empty() : tensor<64x16x4x4xf32>
    %4 = linalg.fill ins(%cst_2 : f32) outs(%3 : tensor<64x16x4x4xf32>) -> tensor<64x16x4x4xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%2, %cst_1 : tensor<64x16x4x4xf32>, tensor<16x16x4x4xf32>) outs(%4 : tensor<64x16x4x4xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %9 = arith.mulf %in, %in_5 : f32
      %10 = arith.addf %out, %9 : f32
      linalg.yield %10 : f32
    } -> tensor<64x16x4x4xf32>
    %cst_3 = arith.constant dense<1.000000e+00> : tensor<16x16x4x4xf32>
    %cst_4 = arith.constant 0.000000e+00 : f32
    %6 = tensor.empty() : tensor<64x16x4x4xf32>
    %7 = linalg.fill ins(%cst_4 : f32) outs(%6 : tensor<64x16x4x4xf32>) -> tensor<64x16x4x4xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%5, %cst_3 : tensor<64x16x4x4xf32>, tensor<16x16x4x4xf32>) outs(%7 : tensor<64x16x4x4xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %9 = arith.mulf %in, %in_5 : f32
      %10 = arith.addf %out, %9 : f32
      linalg.yield %10 : f32
    } -> tensor<64x16x4x4xf32>
    return %8 : tensor<64x16x4x4xf32>
  }
}
