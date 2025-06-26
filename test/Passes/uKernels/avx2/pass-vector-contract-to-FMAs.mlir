// RUN: tpp-opt %s --vector-contract-to-ukernels  --split-input-file  | FileCheck -check-prefix=CHECK %s
// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=3,32,4" --loop-invariant-code-motion --vectorization-pass --hoist-vector-transfer --vector-contract-to-ukernels  --split-input-file  | FileCheck -check-prefix=CHECK1 %s
// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=3,4,1" --loop-invariant-code-motion --vectorization-pass --hoist-vector-transfer --vector-contract-to-ukernels  --split-input-file  | FileCheck -check-prefix=CHECK2 %s
// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=3,32,2" --loop-invariant-code-motion --vectorization-pass --hoist-vector-transfer --vector-contract-to-ukernels  --split-input-file  | FileCheck -check-prefix=CHECKBF16 %s
// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=3,32,4" --loop-invariant-code-motion --vectorization-pass --hoist-vector-transfer --vector-contract-to-ukernels  --split-input-file  | FileCheck -check-prefix=CHECKBF161 %s
// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=3,4,2" --loop-invariant-code-motion --vectorization-pass --hoist-vector-transfer --vector-contract-to-ukernels  --split-input-file  | FileCheck -check-prefix=CHECKBF162 %s


#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
module {
  func.func @optimal_register_allocation(%arg0: memref<32x24x32xf32>, %arg1: memref<32x32x64xf32>, %arg2: memref<24x64xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c24 = arith.constant 24 : index
    %c3 = arith.constant 3 : index
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %c24 step %c3 {
      scf.for %arg4 = %c0 to %c64 step %c32 {
        %subview = memref.subview %arg2[%arg3, %arg4] [3, 32] [1, 1] : memref<24x64xf32> to memref<3x32xf32, strided<[64, 1], offset: ?>>
        %0 = vector.transfer_read %subview[%c0, %c0], %cst {in_bounds = [true, true]} : memref<3x32xf32, strided<[64, 1], offset: ?>>, vector<3x32xf32>
        %1 = scf.for %arg5 = %c0 to %c32 step %c1 iter_args(%arg6 = %0) -> (vector<3x32xf32>) {
          %2 = scf.for %arg7 = %c0 to %c32 step %c1 iter_args(%arg8 = %arg6) -> (vector<3x32xf32>) {
            %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg7] [1, 3, 1] [1, 1, 1] : memref<32x24x32xf32> to memref<1x3x1xf32, strided<[768, 32, 1], offset: ?>>
            %subview_1 = memref.subview %arg1[%arg5, %arg7, %arg4] [1, 1, 32] [1, 1, 1] : memref<32x32x64xf32> to memref<1x1x32xf32, strided<[2048, 64, 1], offset: ?>>
            %3 = vector.transfer_read %subview_0[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x3x1xf32, strided<[768, 32, 1], offset: ?>>, vector<1x3x1xf32>
            %4 = vector.transfer_read %subview_1[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x1x32xf32, strided<[2048, 64, 1], offset: ?>>, vector<1x1x32xf32>
            %5 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %3, %4, %arg8 : vector<1x3x1xf32>, vector<1x1x32xf32> into vector<3x32xf32>
            scf.yield %5 : vector<3x32xf32>
          }
          scf.yield %2 : vector<3x32xf32>
        }
        vector.transfer_write %1, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<3x32xf32>, memref<3x32xf32, strided<[64, 1], offset: ?>>
      }
    }
    return
  }
}

// CHECK-LABEL:   func.func @optimal_register_allocation
// CHECK: scf.for
// CHECK-NEXT: scf.for
// CHECK-NEXT: scf.for
// CHECK-NEXT: scf.for
// CHECK-NEXT: memref.subview
// CHECK-NEXT: memref.subview
// CHECK-NEXT: memref.subview
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.broadcast
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.broadcast
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.broadcast
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.broadcast
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.fma
// CHECK-NEXT: vector.fma
// CHECK-NEXT: vector.fma
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.fma
// CHECK-NEXT: vector.fma
// CHECK-NEXT: vector.fma
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.fma
// CHECK-NEXT: vector.fma
// CHECK-NEXT: vector.fma
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.fma
// CHECK-NEXT: vector.fma
// CHECK-NEXT: vector.fma

// -----

module {
 func.func @No_lowering(%arg0: memref<32x24x32xf32>, %arg1: memref<32x32x64xf32>, %arg2: memref<24x64xf32>) {
     linalg.batch_reduce_matmul ins(%arg0, %arg1 : memref<32x24x32xf32>, memref<32x32x64xf32>) outs(%arg2 : memref<24x64xf32>)
   return
 }
}


// CHECK1-LABEL: @No_lowering
// CHECK1-NOT: vector.fma
// CHECK1: vector.contract

// CHECK2-LABEL: @No_lowering
// CHECK2-NOT: vector.fma
// CHECK2: vector.contract


// -----


module {
 memref.global "private" constant @__constant_2x16x128x2xbf16 : memref<2x16x128x2xbf16> = dense<1.000000e+00> {alignment = 64 : i64}
 func.func @optimal_register_allocation_bf16(%arg0: memref<2x24x16x2xbf16>) -> memref<24x128xbf16> {
   %0 = memref.get_global @__constant_2x16x128x2xbf16 : memref<2x16x128x2xbf16>
   %alloc = memref.alloc() {alignment = 64 : i64} : memref<24x128xbf16>
   %cst = arith.constant 0.000000e+00 : bf16
   linalg.fill ins(%cst : bf16) outs(%alloc : memref<24x128xbf16>)
 
     linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>, affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %0 : memref<2x24x16x2xbf16>, memref<2x16x128x2xbf16>) outs(%alloc : memref<24x128xbf16>) {
     ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
       %1 = arith.mulf %in, %in_1 : bf16
       %2 = arith.addf %out, %1 : bf16
       linalg.yield %2 : bf16
     }
   return %alloc : memref<24x128xbf16>
 }
}

// CHECKBF16-LABEL: @optimal_register_allocation_bf16
// CHECKBF16-COUNT-24: vector.fma

// CHECKBF161-LABEL: @optimal_register_allocation_bf16
// CHECKBF161-NOT: vector.fma
// CHECKBF161: vector.contract

// CHECKBF162-LABEL: @optimal_register_allocation
// CHECKBF162-NOT: vector.fma
// CHECKBF162: vector.contract
