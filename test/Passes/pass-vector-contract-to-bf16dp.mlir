// RUN: tpp-opt %s --vector-contract-to-bf16dp --split-input-file  | FileCheck -check-prefix=CONF1 %s
// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=2,32,2" --loop-invariant-code-motion --vectorization-pass --vector-contract-to-bf16dp --split-input-file  | FileCheck -check-prefix=CONF2 %s
// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=4,32,2" --loop-invariant-code-motion --vectorization-pass --vector-contract-to-bf16dp --split-input-file  | FileCheck -check-prefix=CONF3 %s
// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=2,16,2" --loop-invariant-code-motion --vectorization-pass --vector-contract-to-bf16dp --split-input-file  | FileCheck -check-prefix=CONF4 %s
// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=2,32,4" --loop-invariant-code-motion --vectorization-pass --vector-contract-to-bf16dp --split-input-file  | FileCheck -check-prefix=CONF5 %s

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2)>
module {
  memref.global "private" constant @__constant_32x16x32x2xbf16 : memref<32x16x32x2xbf16> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func @gemm_lower_to_bf16dp(%arg0: memref<8x32x32x32xbf16>) -> memref<8x32x32x32xbf16> {
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant dense<0.000000e+00> : vector<32x32xbf16>
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_32x16x32x2xbf16 : memref<32x16x32x2xbf16>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x32x32x32xbf16>
    %expand_shape = memref.expand_shape %arg0 [[0], [1], [2], [3, 4]] output_shape [8, 32, 32, 16, 2] : memref<8x32x32x32xbf16> into memref<8x32x32x16x2xbf16>
    scf.forall (%arg1, %arg2) in (8, 32) {
      %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
      vector.transfer_write %cst_0, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<32x32xbf16>, memref<32x32xbf16, strided<[32, 1], offset: ?>>
      %subview_1 = memref.subview %expand_shape[%arg1, 0, 0, 0, 0] [1, 32, 32, 16, 2] [1, 1, 1, 1, 1] : memref<8x32x32x16x2xbf16> to memref<32x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>
      scf.for %arg3 = %c0 to %c32 step %c1 {
        scf.for %arg4 = %c0 to %c32 step %c32 {
          %subview_2 = memref.subview %subview[%arg3, %arg4] [1, 32] [1, 1] : memref<32x32xbf16, strided<[32, 1], offset: ?>> to memref<1x32xbf16, strided<[32, 1], offset: ?>>
          scf.for %arg5 = %c0 to %c32 step %c1 {
            scf.for %arg6 = %c0 to %c16 step %c1 {
              %subview_3 = memref.subview %subview_1[%arg5, %arg3, %arg6, 0] [1, 1, 1, 2] [1, 1, 1, 1] : memref<32x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>> to memref<1x1x1x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>
              %subview_4 = memref.subview %0[%arg5, %arg6, %arg4, 0] [1, 1, 32, 2] [1, 1, 1, 1] : memref<32x16x32x2xbf16> to memref<1x1x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
              %1 = vector.transfer_read %subview_3[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x1x1x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>, vector<1x1x1x2xbf16>
              %2 = vector.transfer_read %subview_4[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x1x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>, vector<1x1x32x2xbf16>
              %3 = vector.transfer_read %subview_2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x32xbf16, strided<[32, 1], offset: ?>>, vector<1x32xbf16>
              %4 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction", "reduction"], kind = #vector.kind<add>} %1, %2, %3 : vector<1x1x1x2xbf16>, vector<1x1x32x2xbf16> into vector<1x32xbf16>
              vector.transfer_write %4, %subview_2[%c0, %c0] {in_bounds = [true, true]} : vector<1x32xbf16>, memref<1x32xbf16, strided<[32, 1], offset: ?>>
            }
          }
        }
      }
    }
    return %alloc : memref<8x32x32x32xbf16>
  }
}

// CONF1-LABEL:   memref.global "private" constant @__constant_32x16x32x2xbf16 : memref<32x16x32x2xbf16> = dense<1.000000e+00> {alignment = 64 : i64}
// CONF1-LABEL:   func.func @gemm_lower_to_bf16dp(
// CONF1-SAME:                     %[[VAL_0:.*]]: memref<8x32x32x32xbf16>) -> memref<8x32x32x32xbf16> {
// CONF1:           %[[VAL_1:.*]] = arith.constant 16 : i32
// CONF1:           %[[VAL_2:.*]] = arith.constant dense<0.000000e+00> : vector<32x32xbf16>
// CONF1:           %[[VAL_3:.*]] = arith.constant 16 : index
// CONF1:           %[[VAL_4:.*]] = arith.constant 1 : index
// CONF1:           %[[VAL_5:.*]] = arith.constant 32 : index
// CONF1:           %[[VAL_6:.*]] = arith.constant 0 : index
// CONF1:           %[[VAL_7:.*]] = memref.get_global @__constant_32x16x32x2xbf16 : memref<32x16x32x2xbf16>
// CONF1:           %[[VAL_8:.*]] = memref.alloc() {alignment = 64 : i64} : memref<8x32x32x32xbf16>
// CONF1:           %[[VAL_9:.*]] = memref.expand_shape %[[VAL_0]] {{\[\[}}0], [1], [2], [3, 4]] output_shape [8, 32, 32, 16, 2] : memref<8x32x32x32xbf16> into memref<8x32x32x16x2xbf16>
// CONF1:           scf.forall (%[[VAL_10:.*]], %[[VAL_11:.*]]) in (8, 32) {
// CONF1:             %[[VAL_12:.*]] = memref.subview %[[VAL_8]]{{\[}}%[[VAL_10]], %[[VAL_11]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
// CONF1:             vector.transfer_write %[[VAL_2]], %[[VAL_12]]{{\[}}%[[VAL_6]], %[[VAL_6]]] {in_bounds = [true, true]} : vector<32x32xbf16>, memref<32x32xbf16, strided<[32, 1], offset: ?>>
// CONF1:             %[[VAL_13:.*]] = memref.subview %[[VAL_9]]{{\[}}%[[VAL_10]], 0, 0, 0, 0] [1, 32, 32, 16, 2] [1, 1, 1, 1, 1] : memref<8x32x32x16x2xbf16> to memref<32x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>
// CONF1:             scf.for %[[VAL_14:.*]] = %[[VAL_6]] to %[[VAL_5]] step %[[VAL_4]] {
// CONF1:               scf.for %[[VAL_15:.*]] = %[[VAL_6]] to %[[VAL_5]] step %[[VAL_5]] {
// CONF1:                 %[[VAL_16:.*]] = memref.subview %[[VAL_12]]{{\[}}%[[VAL_14]], %[[VAL_15]]] [1, 32] [1, 1] : memref<32x32xbf16, strided<[32, 1], offset: ?>> to memref<1x32xbf16, strided<[32, 1], offset: ?>>
// CONF1:                 %[[VAL_17:.*]] = memref.subview %[[VAL_16]][0, 0] [1, 32] [1, 1] : memref<1x32xbf16, strided<[32, 1], offset: ?>> to memref<1x32xbf16, strided<[32, 1], offset: ?>>
// CONF1:                 %[[VAL_18:.*]] = vector.load %[[VAL_17]]{{\[}}%[[VAL_6]], %[[VAL_6]]] : memref<1x32xbf16, strided<[32, 1], offset: ?>>, vector<16xbf16>
// CONF1:                 %[[VAL_19:.*]] = vector.bitcast %[[VAL_18]] : vector<16xbf16> to vector<16xi16>
// CONF1:                 %[[VAL_20:.*]] = arith.extui %[[VAL_19]] : vector<16xi16> to vector<16xi32>
// CONF1:                 %[[VAL_21:.*]] = vector.broadcast %[[VAL_1]] : i32 to vector<16xi32>
// CONF1:                 %[[VAL_22:.*]] = arith.shli %[[VAL_20]], %[[VAL_21]] : vector<16xi32>
// CONF1:                 %[[VAL_23:.*]] = vector.bitcast %[[VAL_22]] : vector<16xi32> to vector<16xf32>
// CONF1:                 %[[VAL_24:.*]] = vector.load %[[VAL_17]]{{\[}}%[[VAL_6]], %[[VAL_3]]] : memref<1x32xbf16, strided<[32, 1], offset: ?>>, vector<16xbf16>
// CONF1:                 %[[VAL_25:.*]] = vector.bitcast %[[VAL_24]] : vector<16xbf16> to vector<16xi16>
// CONF1:                 %[[VAL_26:.*]] = arith.extui %[[VAL_25]] : vector<16xi16> to vector<16xi32>
// CONF1:                 %[[VAL_27:.*]] = vector.broadcast %[[VAL_1]] : i32 to vector<16xi32>
// CONF1:                 %[[VAL_28:.*]] = arith.shli %[[VAL_26]], %[[VAL_27]] : vector<16xi32>
// CONF1:                 %[[VAL_29:.*]] = vector.bitcast %[[VAL_28]] : vector<16xi32> to vector<16xf32>
// CONF1:                 %[[VAL_30:.*]]:2 = scf.for %[[VAL_31:.*]] = %[[VAL_6]] to %[[VAL_5]] step %[[VAL_4]] iter_args(%[[VAL_32:.*]] = %[[VAL_23]], %[[VAL_33:.*]] = %[[VAL_29]]) -> (vector<16xf32>, vector<16xf32>) {
// CONF1:                   %[[VAL_34:.*]]:2 = scf.for %[[VAL_35:.*]] = %[[VAL_6]] to %[[VAL_3]] step %[[VAL_4]] iter_args(%[[VAL_36:.*]] = %[[VAL_32]], %[[VAL_37:.*]] = %[[VAL_33]]) -> (vector<16xf32>, vector<16xf32>) {
// CONF1:                     %[[VAL_38:.*]] = memref.subview %[[VAL_13]]{{\[}}%[[VAL_31]], %[[VAL_14]], %[[VAL_35]], 0] [1, 1, 1, 2] [1, 1, 1, 1] : memref<32x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>> to memref<1x1x1x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>
// CONF1:                     %[[VAL_39:.*]] = vector.load %[[VAL_38]]{{\[}}%[[VAL_6]], %[[VAL_6]], %[[VAL_6]], %[[VAL_6]]] : memref<1x1x1x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>, vector<2xbf16>
// CONF1:                     %[[VAL_40:.*]] = vector.bitcast %[[VAL_39]] : vector<2xbf16> to vector<1xi32>
// CONF1:                     %[[VAL_41:.*]] = vector.broadcast %[[VAL_40]] : vector<1xi32> to vector<16xi32>
// CONF1:                     %[[VAL_42:.*]] = vector.bitcast %[[VAL_41]] : vector<16xi32> to vector<32xbf16>
// CONF1:                     %[[VAL_43:.*]] = memref.subview %[[VAL_7]]{{\[}}%[[VAL_31]], %[[VAL_35]], %[[VAL_15]], 0] [1, 1, 32, 2] [1, 1, 1, 1] : memref<32x16x32x2xbf16> to memref<1x1x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
// CONF1:                     %[[VAL_44:.*]] = vector.load %[[VAL_43]]{{\[}}%[[VAL_6]], %[[VAL_6]], %[[VAL_6]], %[[VAL_6]]] : memref<1x1x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>, vector<32xbf16>
// CONF1:                     %[[VAL_45:.*]] = vector.load %[[VAL_43]]{{\[}}%[[VAL_6]], %[[VAL_6]], %[[VAL_3]], %[[VAL_6]]] : memref<1x1x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>, vector<32xbf16>
// CONF1:                     %[[VAL_46:.*]] = x86vector.avx512.dot %[[VAL_36]], %[[VAL_42]], %[[VAL_44]] : vector<32xbf16> -> vector<16xf32>
// CONF1:                     %[[VAL_47:.*]] = x86vector.avx512.dot %[[VAL_37]], %[[VAL_42]], %[[VAL_45]] : vector<32xbf16> -> vector<16xf32>
// CONF1:                     scf.yield %[[VAL_46]], %[[VAL_47]] : vector<16xf32>, vector<16xf32>
// CONF1:                   }
// CONF1:                   scf.yield %[[VAL_48:.*]]#0, %[[VAL_48]]#1 : vector<16xf32>, vector<16xf32>
// CONF1:                 }
// CONF1:                 %[[VAL_49:.*]] = arith.truncf %[[VAL_50:.*]]#0 : vector<16xf32> to vector<16xbf16>
// CONF1:                 vector.store %[[VAL_49]], %[[VAL_17]]{{\[}}%[[VAL_6]], %[[VAL_6]]] : memref<1x32xbf16, strided<[32, 1], offset: ?>>, vector<16xbf16>
// CONF1:                 %[[VAL_51:.*]] = arith.truncf %[[VAL_50]]#1 : vector<16xf32> to vector<16xbf16>
// CONF1:                 vector.store %[[VAL_51]], %[[VAL_17]]{{\[}}%[[VAL_6]], %[[VAL_3]]] : memref<1x32xbf16, strided<[32, 1], offset: ?>>, vector<16xbf16>
// CONF1:               }
// CONF1:             }
// CONF1:           }
// CONF1:           return %[[VAL_8]] : memref<8x32x32x32xbf16>
// CONF1:         }


// -----


#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
module {
  memref.global "private" constant @__constant_16x32x64x2xbf16 : memref<16x32x64x2xbf16> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func @gemm_64_tiles_testing_different_cases(%arg0: memref<4x16x64x64xbf16>) -> memref<4x16x64x64xbf16> {
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


// CONF2-LABEL: func.func @gemm_64_tiles_testing_different_cases
// CONF2: x86vector.avx512.dot
// CONF2-NEXT: x86vector.avx512.dot 
// CONF2-NEXT: x86vector.avx512.dot
// CONF2-NEXT: x86vector.avx512.dot
// CONF2-NEXT: scf.yield

// CONF3-LABEL: func.func @gemm_64_tiles_testing_different_cases
// CONF3: x86vector.avx512.dot
// CONF3-NEXT: x86vector.avx512.dot
// CONF3-NEXT: x86vector.avx512.dot
// CONF3-NEXT: x86vector.avx512.dot
// CONF3-NEXT: x86vector.avx512.dot
// CONF3-NEXT: x86vector.avx512.dot
// CONF3-NEXT: x86vector.avx512.dot
// CONF3-NEXT: x86vector.avx512.dot
// CONF3-NEXT: scf.yield

// CONF4-LABEL: func.func @gemm_64_tiles_testing_different_cases
// CONF4-NOT: x86vector.avx512.dot

// CONF5-LABEL: func.func @gemm_64_tiles_testing_different_cases
// CONF5-NOT: x86vector.avx512.dot

// -----

module {
  func.func @gemm_no_bf16dp_lowering(%arg0: memref<16x32x16x32xf32>, %arg1: memref<32x32x32x32xf32>, %arg2: memref<16x32x16x32xf32>) {
    scf.forall (%arg3, %arg4) in (16, 32) {
      %subview = memref.subview %arg0[%arg3, 0, 0, 0] [1, 32, 16, 32] [1, 1, 1, 1] : memref<16x32x16x32xf32> to memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[%arg4, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<32x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      %subview_1 = memref.subview %arg2[%arg3, %arg4, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<16x32x16x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
      linalg.batch_reduce_matmul ins(%subview, %subview_0 : memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>>, memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%subview_1 : memref<16x32xf32, strided<[32, 1], offset: ?>>)
    }
    return
  }
}

// CONF2-LABEL: func.func @gemm_no_bf16dp_lowering
// CONF2-NOT: x86vector.avx512.dot

// CONF3-LABEL: func.func @gemm_no_bf16dp_lowering
// CONF3-NOT: x86vector.avx512.dot
