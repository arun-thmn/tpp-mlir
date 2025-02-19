// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=8,32,1" --split-input-file  | FileCheck -check-prefix=CONF1 %s
// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=32,32,32" --split-input-file  | FileCheck -check-prefix=CONF2 %s

module {
  func.func @gemm_do_register_tiling(%arg0: memref<16x32x16x32xf32>, %arg1: memref<32x32x32x32xf32>, %arg2: memref<16x32x16x32xf32>) {
    scf.forall (%arg3, %arg4) in (16, 32) {
      %subview = memref.subview %arg0[%arg3, 0, 0, 0] [1, 32, 16, 32] [1, 1, 1, 1] : memref<16x32x16x32xf32> to memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[%arg4, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<32x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      %subview_1 = memref.subview %arg2[%arg3, %arg4, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<16x32x16x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
      linalg.batch_reduce_matmul ins(%subview, %subview_0 : memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>>, memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%subview_1 : memref<16x32xf32, strided<[32, 1], offset: ?>>)
    }
    return
  }
}


// CONF1-LABEL: func.func @gemm_do_register_tiling
// CONF1-DAG: %[[C1:.+]] = arith.constant 1 : index
// CONF1-DAG: %[[C32:.+]] = arith.constant 32 : index
// CONF1-DAG: %[[C8:.+]] = arith.constant 8 : index
// CONF1-DAG: %[[C16:.+]] = arith.constant 16 : index
// CONF1-DAG: %[[C0:.+]] = arith.constant 0 : index
// CONF1: scf.forall (%arg3, %arg4) in (16, 32) {
// CONF1-NEXT: %subview = memref.subview %arg0[%arg3, 0, 0, 0] [1, 32, 16, 32] [1, 1, 1, 1] : memref<16x32x16x32xf32> to memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>>
// CONF1-NEXT: %subview_0 = memref.subview %arg1[%arg4, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<32x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CONF1-NEXT: %subview_1 = memref.subview %arg2[%arg3, %arg4, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<16x32x16x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
// CONF1-NEXT: scf.for %[[I:.+]] = %[[C0]] to %[[C16]] step %[[C8]] {
// CONF1-NEXT:  scf.for %[[J:.+]] = %[[C0]] to %[[C32]] step %[[C32]] {
// CONF1-NEXT:   scf.for %[[K:.+]] = %[[C0]] to %[[C32]] step %[[C1]] {
// CONF1-NEXT:    scf.for %[[L:.+]] = %[[C0]] to %[[C32]] step %[[C1]] {
// CONF1-NEXT:     %subview_2 = memref.subview %subview[%[[K]], %[[I]], %[[L]]] [1, 8, 1] [1, 1, 1] : memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>> to memref<1x8x1xf32, strided<[512, 32, 1], offset: ?>>
// CONF1-NEXT:     %subview_3 = memref.subview %subview_0[%[[K]], %[[L]], %[[J]]] [1, 1, 32] [1, 1, 1] : memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<1x1x32xf32, strided<[1024, 32, 1], offset: ?>>
// CONF1-NEXT:     %subview_4 = memref.subview %subview_1[%[[I]], %[[J]]] [8, 32] [1, 1] : memref<16x32xf32, strided<[32, 1], offset: ?>> to memref<8x32xf32, strided<[32, 1], offset: ?>>
// CONF1-NEXT:     linalg.batch_reduce_matmul ins(%subview_2, %subview_3 : memref<1x8x1xf32, strided<[512, 32, 1], offset: ?>>, memref<1x1x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%subview_4 : memref<8x32xf32, strided<[32, 1], offset: ?>>)

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
module {
  memref.global "private" constant @__constant_32x16x32x2xbf16 : memref<32x16x32x2xbf16> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func @gemm_32tiles_do_tiling_bf16(%arg0: memref<8x32x32x32xbf16>) -> memref<8x32x32x32xbf16> {
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
}

// CONF2-LABEL: func.func @gemm_32tiles_do_tiling_bf16
// CONF2-DAG: %[[C1:.+]] = arith.constant 1 : index
// CONF2-DAG: %[[C32:.+]] = arith.constant 32 : index
// CONF2-DAG: %[[C16:.+]] = arith.constant 16 : index
// CONF2-DAG: %[[C0:.+]] = arith.constant 0 : index
// CONF2: %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
// CONF2-NEXT: linalg.fill ins(%cst : bf16) outs(%subview : memref<32x32xbf16, strided<[32, 1], offset: ?>>)
// CONF2-NEXT: %subview_0 = memref.subview %expand_shape[%arg1, 0, 0, 0, 0] [1, 32, 32, 16, 2] [1, 1, 1, 1, 1] : memref<8x32x32x16x2xbf16> to memref<32x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>
// CONF2-NEXT:  scf.for %[[I:.+]] = %[[C0]] to %[[C32]] step %[[C32]] {
// CONF2-NEXT:   scf.for %[[J:.+]] = %[[C0]] to %[[C32]] step %[[C32]] {
// CONF2-NEXT:    scf.for %[[K:.+]] = %[[C0]] to %[[C32]] step %[[C1]] {
// CONF2-NEXT:     scf.for %[[L:.+]] = %[[C0]] to %[[C16]] step %[[C16]] {
// CONF2-NEXT:      %subview_1 = memref.subview %subview_0[%[[K]], %[[I]], %[[L]], 0] [1, 32, 16, 2] [1, 1, 1, 1] : memref<32x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>> to memref<1x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>
// CONF2-NEXT:      %subview_2 = memref.subview %0[%[[K]], %[[L]], %[[J]], 0]  [1, 16, 32, 2] [1, 1, 1, 1] : memref<32x16x32x2xbf16> to memref<1x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
// CONF2-NEXT:      %subview_3 = memref.subview %subview[%[[I]], %[[J]]]  [32, 32] [1, 1] : memref<32x32xbf16, strided<[32, 1], offset: ?>> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
// CONF2-NEXT:      linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%subview_1, %subview_2 : memref<1x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>, memref<1x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>) outs(%subview_3 : memref<32x32xbf16, strided<[32, 1], offset: ?>>)

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d2)>
module {
  memref.global "private" constant @__constant_16x32x64x2xbf16 : memref<16x32x64x2xbf16> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func @gemm_64tiles_do_tiling_bf16(%arg0: memref<4x16x64x64xbf16>) -> memref<4x16x64x64xbf16> {
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = memref.get_global @__constant_16x32x64x2xbf16 : memref<16x32x64x2xbf16>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x16x64x64xbf16>
    %expand_shape = memref.expand_shape %arg0 [[0], [1], [2], [3, 4]] output_shape [4, 16, 64, 32, 2] : memref<4x16x64x64xbf16> into memref<4x16x64x32x2xbf16>
    scf.forall (%arg1, %arg2) in (4, 16) {
      %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : memref<4x16x64x64xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
      linalg.fill ins(%cst : bf16) outs(%subview : memref<64x64xbf16, strided<[64, 1], offset: ?>>)
      %subview_0 = memref.subview %expand_shape[%arg1, 0, 0, 0, 0] [1, 16, 64, 32, 2] [1, 1, 1, 1, 1] : memref<4x16x64x32x2xbf16> to memref<16x64x32x2xbf16, strided<[4096, 64, 2, 1], offset: ?>>
      linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%subview_0, %0 : memref<16x64x32x2xbf16, strided<[4096, 64, 2, 1], offset: ?>>, memref<16x32x64x2xbf16>) outs(%subview : memref<64x64xbf16, strided<[64, 1], offset: ?>>) {
      ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
        %1 = arith.mulf %in, %in_1 : bf16
        %2 = arith.addf %out, %1 : bf16
        linalg.yield %2 : bf16
      }
    }
    return %alloc : memref<4x16x64x64xbf16>
  }
}

// CONF2-LABEL: func.func @gemm_64tiles_do_tiling_bf16
// CONF2-DAG: %[[C1:.+]] = arith.constant 1 : index
// CONF2-DAG: %[[C32:.+]] = arith.constant 32 : index
// CONF2-DAG: %[[C64:.+]] = arith.constant 64 : index
// CONF2-DAG: %[[C16:.+]] = arith.constant 16 : index
// CONF2-DAG: %[[C0:.+]] = arith.constant 0 : index
// CONF2: %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : memref<4x16x64x64xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
// CONF2-NEXT: linalg.fill ins(%cst : bf16) outs(%subview : memref<64x64xbf16, strided<[64, 1], offset: ?>>)
// CONF2-NEXT: %subview_0 = memref.subview %expand_shape[%arg1, 0, 0, 0, 0] [1, 16, 64, 32, 2] [1, 1, 1, 1, 1] : memref<4x16x64x32x2xbf16> to memref<16x64x32x2xbf16, strided<[4096, 64, 2, 1], offset: ?>>
// CONF2-NEXT: scf.for %[[I:.+]] = %[[C0]] to %[[C64]] step %[[C32]] {
// CONF2-NEXT:  scf.for %[[J:.+]] = %[[C0]] to %[[C64]] step %[[C32]] {
// CONF2-NEXT:   scf.for %[[K:.+]] = %[[C0]] to %[[C16]] step %[[C1]] {
// CONF2-NEXT:    scf.for %[[L:.+]] = %[[C0]] to %[[C32]] step %[[C16]] {
// CONF2-NEXT:     %subview_1 = memref.subview %subview_0[%[[K]], %[[I]], %[[L]], 0] [1, 32, 16, 2] [1, 1, 1, 1] : memref<16x64x32x2xbf16, strided<[4096, 64, 2, 1], offset: ?>> to memref<1x32x16x2xbf16, strided<[4096, 64, 2, 1], offset: ?>>
// CONF2-NEXT:     %subview_2 = memref.subview %0[%[[K]], %[[L]], %[[J]], 0] [1, 16, 32, 2] [1, 1, 1, 1] : memref<16x32x64x2xbf16> to memref<1x16x32x2xbf16, strided<[4096, 128, 2, 1], offset: ?>>
// CONF2-NEXT:     %subview_3 = memref.subview %subview[%[[I]], %[[J]]] [32, 32] [1, 1] : memref<64x64xbf16, strided<[64, 1], offset: ?>> to memref<32x32xbf16, strided<[64, 1], offset: ?>>
// CONF2-NEXT:     linalg.generic

// -----

module {
  func.func @brgemm_tensor_type_no_tiling(%arg0: tensor<128x256x512xf32>, %arg1: tensor<128x512x256xf32>, %arg2: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %0 = linalg.batch_reduce_matmul ins(%arg0, %arg1 : tensor<128x256x512xf32>, tensor<128x512x256xf32>) outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
    return %0 : tensor<256x256xf32>
  }
}


// CONF1-LABEL: func.func @brgemm_tensor_type_no_tiling
// CONF1-NOT: scf.for
// CONF2-NOT: scf.for

// -----

module {
  func.func @matmul_no_tiling(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
     linalg.matmul ins(%arg0, %arg1 : memref<64x64xf32>, memref<64x64xf32>)
                outs(%arg2 : memref<64x64xf32>)
     return
  }
}


// CONF1-LABEL: func.func @matmul_no_tiling
// CONF1-NOT: scf.for
// CONF2-NOT: scf.for
