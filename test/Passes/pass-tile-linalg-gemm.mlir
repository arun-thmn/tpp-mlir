//RUN: ./tpp-opt --tile-consumer-and-fuse-producers --bufferize --linalg-generalize-named-ops  --tile-linalg="mTile=4,8 nTile=8,16" ./pass-tile-linalg-gemm.mlir



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
