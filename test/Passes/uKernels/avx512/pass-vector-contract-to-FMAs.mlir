// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=6,64,1" --loop-invariant-code-motion --vectorization-pass --hoist-vector-transfer --vector-contract-to-micro-kernels  --split-input-file  | FileCheck -check-prefix=CHECK %s
// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=6,8,1" --loop-invariant-code-motion --vectorization-pass --hoist-vector-transfer --vector-contract-to-micro-kernels  --split-input-file  | FileCheck -check-prefix=CHECK1 %s
// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=6,64,4" --loop-invariant-code-motion --vectorization-pass --hoist-vector-transfer --vector-contract-to-micro-kernels  --split-input-file  | FileCheck -check-prefix=CHECK2 %s

module {
 func.func @optimal_register_allocation(%arg0: memref<32x24x32xf32>, %arg1: memref<32x32x64xf32>, %arg2: memref<24x64xf32>) -> memref<24x64xf32> {
     linalg.batch_reduce_matmul ins(%arg0, %arg1 : memref<32x24x32xf32>, memref<32x32x64xf32>) outs(%arg2 : memref<24x64xf32>)
   return %arg2 : memref<24x64xf32>
 }
}

// CHECK-LABEL:   func.func @optimal_register_allocation
// CHECK: vector.fma
// CHECK-NEXT: vector.fma
// CHECK-NEXT: vector.fma
// CHECK-NEXT: vector.fma
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.broadcast
// CHECK-NEXT: vector.fma
// CHECK-NEXT: vector.fma
// CHECK-NEXT: vector.fma
// CHECK-NEXT: vector.fma
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.broadcast
// CHECK-NEXT: vector.fma
// CHECK-NEXT: vector.fma
// CHECK-NEXT: vector.fma
// CHECK-NEXT: vector.fma
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.broadcast
// CHECK-NEXT: vector.fma
// CHECK-NEXT: vector.fma
// CHECK-NEXT: vector.fma
// CHECK-NEXT: vector.fma
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.broadcast
// CHECK-NEXT: vector.fma
// CHECK-NEXT: vector.fma
// CHECK-NEXT: vector.fma
// CHECK-NEXT: vector.fma
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.broadcast
// CHECK-NEXT: vector.fma
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
