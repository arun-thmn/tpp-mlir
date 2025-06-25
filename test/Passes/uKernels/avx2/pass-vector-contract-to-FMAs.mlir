// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=3,32,1" --loop-invariant-code-motion --vectorization-pass --hoist-vector-transfer --vector-contract-to-ukernels  --split-input-file  | FileCheck -check-prefix=CHECK %s
// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=3,32,4" --loop-invariant-code-motion --vectorization-pass --hoist-vector-transfer --vector-contract-to-ukernels  --split-input-file  | FileCheck -check-prefix=CHECK1 %s
// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=3,4,1" --loop-invariant-code-motion --vectorization-pass --hoist-vector-transfer --vector-contract-to-ukernels  --split-input-file  | FileCheck -check-prefix=CHECK2 %s
// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=3,32,2" --loop-invariant-code-motion --vectorization-pass --hoist-vector-transfer --vector-contract-to-ukernels  --split-input-file  | FileCheck -check-prefix=CHECKBF16 %s
// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=3,32,4" --loop-invariant-code-motion --vectorization-pass --hoist-vector-transfer --vector-contract-to-ukernels  --split-input-file  | FileCheck -check-prefix=CHECKBF161 %s
// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=3,4,2" --loop-invariant-code-motion --vectorization-pass --hoist-vector-transfer --vector-contract-to-ukernels  --split-input-file  | FileCheck -check-prefix=CHECKBF162 %s

module {
 func.func @optimal_register_allocation(%arg0: memref<32x24x32xf32>, %arg1: memref<32x32x64xf32>, %arg2: memref<24x64xf32>) {
     linalg.batch_reduce_matmul ins(%arg0, %arg1 : memref<32x24x32xf32>, memref<32x32x64xf32>) outs(%arg2 : memref<24x64xf32>)
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


// CHECK1-LABEL: @optimal_register_allocation
// CHECK1-NOT: vector.fma
// CHECK1: vector.contract

// CHECK2-LABEL: @optimal_register_allocation
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
