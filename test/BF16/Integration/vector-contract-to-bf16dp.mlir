// RUN: tpp-run  -e gemm_bf16_dp_random_AB --entry-point-result=void -print --splat-to-random --init-type normal  -seed 123  %s > %t.1
// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=4,32,2"  --loop-invariant-code-motion --vectorization-pass --vector-contract-to-bf16dp | tpp-run  -e gemm_bf16_dp_random_AB --entry-point-result=void -print  --splat-to-random --init-type normal  -seed 123  > %t.2
// RUN: fpcmp -r 0.001 %t.1 %t.2

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
memref.global "private" constant @__constant_32x16x32x2xbf16 : memref<32x16x32x2xbf16> = dense<1.000000e+00> {alignment = 64 : i64}
func.func @gemm_bf16_dp_random_AB(%arg0: memref<8x32x32x32xbf16>) -> memref<8x32x32x32xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = memref.get_global @__constant_32x16x32x2xbf16 : memref<32x16x32x2xbf16>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x32x32x32xbf16>
  %expand_shape = memref.expand_shape %arg0 [[0], [1], [2], [3, 4]] output_shape [8, 32, 32, 16, 2] : memref<8x32x32x32xbf16> into memref<8x32x32x16x2xbf16>
  scf.forall (%arg1, %arg2) in (8, 32) {
    %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
    linalg.fill ins(%cst : bf16) outs(%subview : memref<32x32xbf16, strided<[32, 1], offset: ?>>)
    %subview_0 = memref.subview %expand_shape[%arg1, 0, 0, 0, 0] [1, 32, 32, 16, 2] [1, 1, 1, 1, 1] : memref<8x32x32x16x2xbf16> to memref<32x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>
    linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%subview_0, %0 : memref<32x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>, memref<32x16x32x2xbf16>) outs(%subview : memref<32x32xbf16, strided<[32, 1], offset: ?>>) {
    ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
      %1 = arith.mulf %in, %in_1 : bf16
      %2 = arith.addf %out, %1 : bf16
      linalg.yield %2 : bf16
    }
  }
  return %alloc : memref<8x32x32x32xbf16>
}

// RUN: tpp-run -e gemm_bf16_args --entry-point-result=void -print --splat-to-random --init-type normal  -seed 123  %s > %t.1
// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=8,32,2"  --loop-invariant-code-motion --vectorization-pass --vector-contract-to-bf16dp | tpp-run -e  gemm_bf16_args --entry-point-result=void -print  --splat-to-random --init-type normal  -seed 123  > %t.2
// RUN: fpcmp -r 0.01 %t.1 %t.2

#mapA = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#mapB = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#mapC = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
func.func @gemm_bf16_args(%arg0: memref<8x32x32x32xbf16>, %arg1: memref<32x32x16x32x2xbf16>, %arg2: memref<8x32x32x32xbf16>) -> memref<8x32x32x32xbf16> {
  %expand_shape = memref.expand_shape %arg0 [[0], [1], [2], [3, 4]] output_shape [8, 32, 32, 16, 2] : memref<8x32x32x32xbf16> into memref<8x32x32x16x2xbf16>
  scf.forall (%arg3, %arg4) in (8, 32) {
    %subview = memref.subview %expand_shape[%arg3, 0, 0, 0, 0] [1, 32, 32, 16, 2] [1, 1, 1, 1, 1] : memref<8x32x32x16x2xbf16> to memref<32x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>
    %subview_0 = memref.subview %arg1[%arg4, 0, 0, 0, 0] [1, 32, 16, 32, 2] [1, 1, 1, 1, 1] : memref<32x32x16x32x2xbf16> to memref<32x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
    %subview_1 = memref.subview %arg2[%arg3, %arg4, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
    linalg.generic {indexing_maps = [#mapA, #mapB, #mapC], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%subview, %subview_0 : memref<32x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>, memref<32x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>) outs(%subview_1 : memref<32x32xbf16, strided<[32, 1], offset: ?>>) {
    ^bb0(%in: bf16, %in_2: bf16, %out: bf16):
      %0 = arith.mulf %in, %in_2 : bf16
      %1 = arith.addf %out, %0 : bf16
      linalg.yield %1 : bf16
    }
  }
  %alloc = memref.alloc() : memref<8x32x32x32xbf16>
  memref.copy %arg2, %alloc : memref<8x32x32x32xbf16> to memref<8x32x32x32xbf16>
  return %alloc : memref<8x32x32x32xbf16>
}



// RUN: tpp-run -e mlp_bf16 --entry-point-result=void -print --splat-to-random --init-type normal  -seed 123  %s > %t.1
// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=8,32,2"  --loop-invariant-code-motion --vectorization-pass --vector-contract-to-bf16dp | tpp-run -e  mlp_bf16 --entry-point-result=void -print  --splat-to-random --init-type normal  -seed 123  > %t.2
// RUN: fpcmp -r 0.01 %t.1 %t.2


#mapQ = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#mapR = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#mapS = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
#mapT = affine_map<(d0, d1) -> (d1)>
#mapU = affine_map<(d0, d1) -> (d0, d1)>
memref.global "private" constant @__constant_32xbf16 : memref<32xbf16> = dense<1.000000e+00> {alignment = 64 : i64}
func.func @mlp_bf16(%arg0: memref<8x32x32x32xbf16>) -> memref<8x32x32x32xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = memref.get_global @__constant_32xbf16 : memref<32xbf16>
  %1 = memref.get_global @__constant_32x16x32x2xbf16 : memref<32x16x32x2xbf16>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x32x32x32xbf16>
  %expand_shape = memref.expand_shape %arg0 [[0], [1], [2], [3, 4]] output_shape [8, 32, 32, 16, 2] : memref<8x32x32x32xbf16> into memref<8x32x32x16x2xbf16>
  scf.forall (%arg1, %arg2) in (8, 32) {
    %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
    linalg.fill ins(%cst : bf16) outs(%subview : memref<32x32xbf16, strided<[32, 1], offset: ?>>)
    %subview_0 = memref.subview %expand_shape[%arg1, 0, 0, 0, 0] [1, 32, 32, 16, 2] [1, 1, 1, 1, 1] : memref<8x32x32x16x2xbf16> to memref<32x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>
    linalg.generic {indexing_maps = [#mapQ, #mapR, #mapS], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%subview_0, %1 : memref<32x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>, memref<32x16x32x2xbf16>) outs(%subview : memref<32x32xbf16, strided<[32, 1], offset: ?>>) {
    ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
      %2 = arith.mulf %in, %in_1 : bf16
      %3 = arith.addf %out, %2 : bf16
      linalg.yield %3 : bf16
    }
    linalg.generic {indexing_maps = [#mapT, #mapU], iterator_types = ["parallel", "parallel"]} ins(%0 : memref<32xbf16>) outs(%subview : memref<32x32xbf16, strided<[32, 1], offset: ?>>) {
    ^bb0(%in: bf16, %out: bf16):
      %2 = arith.addf %in, %out : bf16
      linalg.yield %2 : bf16
    }
    linalg.generic {indexing_maps = [#mapU], iterator_types = ["parallel", "parallel"]} outs(%subview : memref<32x32xbf16, strided<[32, 1], offset: ?>>) {
    ^bb0(%out: bf16):
      %2 = arith.maximumf %out, %cst : bf16
      linalg.yield %2 : bf16
    }
  }
  return %alloc : memref<8x32x32x32xbf16>
}

