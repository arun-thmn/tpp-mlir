// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=6,64,2" --loop-invariant-code-motion --vectorization-pass --hoist-vector-transfer --vector-contract-to-micro-kernels  --split-input-file  | FileCheck -check-prefix=CHECK %s

module {
 memref.global "private" constant @__constant_2x16x128x2xbf16 : memref<2x16x128x2xbf16> = dense<1.000000e+00> {alignment = 64 : i64}
 func.func @optimal_register_allocation(%arg0: memref<2x24x16x2xbf16>) -> memref<24x128xbf16> {
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

// CHECK-LABEL:   func.func @optimal_register_allocation
// CHECK: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.bitcast
// CHECK-NEXT: vector.broadcast
// CHECK-NEXT: vector.bitcast
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.bitcast
// CHECK-NEXT: vector.broadcast
// CHECK-NEXT: vector.bitcast
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.bitcast
// CHECK-NEXT: vector.broadcast
// CHECK-NEXT: vector.bitcast
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.bitcast
// CHECK-NEXT: vector.broadcast
// CHECK-NEXT: vector.bitcast
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.bitcast
// CHECK-NEXT: vector.broadcast
// CHECK-NEXT: vector.bitcast
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot

// -----
