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

// CHECK-LABEL:   func.func @optimal_register_allocation(
// CHECK-SAME:                     %[[ARG0:.*]]: memref<32x24x32xf32>,
// CHECK-SAME:                     %[[ARG1:.*]]: memref<32x32x64xf32>,
// CHECK-SAME:                     %[[ARG2:.*]]: memref<24x64xf32>) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 8 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 24 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_9:.*]] = arith.constant -65536 : i32
// CHECK:           %[[VAL_10:.*]] = vector.broadcast %[[VAL_9]] : i32 to vector<8xi32>
// CHECK:           %[[VAL_11:.*]] = memref.alloc() : memref<1xvector<8xi32>>
// CHECK:           memref.store %[[VAL_10]], %[[VAL_11]]{{\[}}%[[VAL_3]]] : memref<1xvector<8xi32>>
// CHECK:           scf.for %[[VAL_12:.*]] = %[[VAL_3]] to %[[VAL_4]] step %[[VAL_5]] {
// CHECK:             scf.for %[[VAL_13:.*]] = %[[VAL_3]] to %[[VAL_6]] step %[[VAL_7]] {
// CHECK:               %[[VAL_14:.*]] = memref.subview %[[ARG2]]{{\[}}%[[VAL_12]], %[[VAL_13]]] [3, 32] [1, 1] : memref<24x64xf32> to memref<3x32xf32, strided<[64, 1], offset: ?>>
// CHECK:               %[[VAL_15:.*]] = vector.load %[[VAL_14]]{{\[}}%[[VAL_3]], %[[VAL_3]]] : memref<3x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// CHECK:               %[[VAL_16:.*]] = vector.load %[[VAL_14]]{{\[}}%[[VAL_8]], %[[VAL_3]]] : memref<3x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// CHECK:               %[[VAL_17:.*]] = vector.load %[[VAL_14]]{{\[}}%[[VAL_2]], %[[VAL_3]]] : memref<3x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// CHECK:               %[[VAL_18:.*]] = vector.load %[[VAL_14]]{{\[}}%[[VAL_3]], %[[VAL_1]]] : memref<3x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// CHECK:               %[[VAL_19:.*]] = vector.load %[[VAL_14]]{{\[}}%[[VAL_8]], %[[VAL_1]]] : memref<3x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// CHECK:               %[[VAL_20:.*]] = vector.load %[[VAL_14]]{{\[}}%[[VAL_2]], %[[VAL_1]]] : memref<3x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// CHECK:               %[[VAL_21:.*]] = vector.load %[[VAL_14]]{{\[}}%[[VAL_3]], %[[VAL_0]]] : memref<3x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// CHECK:               %[[VAL_22:.*]] = vector.load %[[VAL_14]]{{\[}}%[[VAL_8]], %[[VAL_0]]] : memref<3x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// CHECK:               %[[VAL_23:.*]] = vector.load %[[VAL_14]]{{\[}}%[[VAL_2]], %[[VAL_0]]] : memref<3x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// CHECK:               %[[VAL_24:.*]] = vector.load %[[VAL_14]]{{\[}}%[[VAL_3]], %[[VAL_4]]] : memref<3x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// CHECK:               %[[VAL_25:.*]] = vector.load %[[VAL_14]]{{\[}}%[[VAL_8]], %[[VAL_4]]] : memref<3x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// CHECK:               %[[VAL_26:.*]] = vector.load %[[VAL_14]]{{\[}}%[[VAL_2]], %[[VAL_4]]] : memref<3x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// CHECK:               %[[VAL_27:.*]]:12 = scf.for %[[VAL_28:.*]] = %[[VAL_3]] to %[[VAL_7]] step %[[VAL_8]] iter_args(%[[VAL_29:.*]] = %[[VAL_15]], %[[VAL_30:.*]] = %[[VAL_16]], %[[VAL_31:.*]] = %[[VAL_17]], %[[VAL_32:.*]] = %[[VAL_18]], %[[VAL_33:.*]] = %[[VAL_19]], %[[VAL_34:.*]] = %[[VAL_20]], %[[VAL_35:.*]] = %[[VAL_21]], %[[VAL_36:.*]] = %[[VAL_22]], %[[VAL_37:.*]] = %[[VAL_23]], %[[VAL_38:.*]] = %[[VAL_24]], %[[VAL_39:.*]] = %[[VAL_25]], %[[VAL_40:.*]] = %[[VAL_26]]) -> (vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>) {
// CHECK:                 %[[VAL_41:.*]]:12 = scf.for %[[VAL_42:.*]] = %[[VAL_3]] to %[[VAL_7]] step %[[VAL_8]] iter_args(%[[VAL_43:.*]] = %[[VAL_29]], %[[VAL_44:.*]] = %[[VAL_30]], %[[VAL_45:.*]] = %[[VAL_31]], %[[VAL_46:.*]] = %[[VAL_32]], %[[VAL_47:.*]] = %[[VAL_33]], %[[VAL_48:.*]] = %[[VAL_34]], %[[VAL_49:.*]] = %[[VAL_35]], %[[VAL_50:.*]] = %[[VAL_36]], %[[VAL_51:.*]] = %[[VAL_37]], %[[VAL_52:.*]] = %[[VAL_38]], %[[VAL_53:.*]] = %[[VAL_39]], %[[VAL_54:.*]] = %[[VAL_40]]) -> (vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>) {
// CHECK:                   %[[VAL_55:.*]] = memref.subview %[[ARG0]]{{\[}}%[[VAL_28]], %[[VAL_12]], %[[VAL_42]]] [1, 3, 1] [1, 1, 1] : memref<32x24x32xf32> to memref<1x3x1xf32, strided<[768, 32, 1], offset: ?>>
// CHECK:                   %[[VAL_56:.*]] = memref.subview %[[ARG1]]{{\[}}%[[VAL_28]], %[[VAL_42]], %[[VAL_13]]] [1, 1, 32] [1, 1, 1] : memref<32x32x64xf32> to memref<1x1x32xf32, strided<[2048, 64, 1], offset: ?>>
// CHECK:                   %[[VAL_57:.*]] = vector.load %[[VAL_55]]{{\[}}%[[VAL_3]], %[[VAL_3]], %[[VAL_3]]] : memref<1x3x1xf32, strided<[768, 32, 1], offset: ?>>, vector<1xf32>
// CHECK:                   %[[VAL_58:.*]] = vector.broadcast %[[VAL_57]] : vector<1xf32> to vector<8xf32>
// CHECK:                   %[[VAL_59:.*]] = vector.load %[[VAL_55]]{{\[}}%[[VAL_3]], %[[VAL_8]], %[[VAL_3]]] : memref<1x3x1xf32, strided<[768, 32, 1], offset: ?>>, vector<1xf32>
// CHECK:                   %[[VAL_60:.*]] = vector.broadcast %[[VAL_59]] : vector<1xf32> to vector<8xf32>
// CHECK:                   %[[VAL_61:.*]] = vector.load %[[VAL_55]]{{\[}}%[[VAL_3]], %[[VAL_2]], %[[VAL_3]]] : memref<1x3x1xf32, strided<[768, 32, 1], offset: ?>>, vector<1xf32>
// CHECK:                   %[[VAL_62:.*]] = vector.broadcast %[[VAL_61]] : vector<1xf32> to vector<8xf32>
// CHECK:                   %[[VAL_63:.*]] = vector.load %[[VAL_56]]{{\[}}%[[VAL_3]], %[[VAL_3]], %[[VAL_3]]] : memref<1x1x32xf32, strided<[2048, 64, 1], offset: ?>>, vector<8xf32>
// CHECK:                   %[[VAL_64:.*]] = vector.fma %[[VAL_58]], %[[VAL_63]], %[[VAL_43]] : vector<8xf32>
// CHECK:                   %[[VAL_65:.*]] = vector.fma %[[VAL_60]], %[[VAL_63]], %[[VAL_44]] : vector<8xf32>
// CHECK:                   %[[VAL_66:.*]] = vector.fma %[[VAL_62]], %[[VAL_63]], %[[VAL_45]] : vector<8xf32>
// CHECK:                   %[[VAL_67:.*]] = vector.load %[[VAL_56]]{{\[}}%[[VAL_3]], %[[VAL_3]], %[[VAL_1]]] : memref<1x1x32xf32, strided<[2048, 64, 1], offset: ?>>, vector<8xf32>
// CHECK:                   %[[VAL_68:.*]] = vector.fma %[[VAL_58]], %[[VAL_67]], %[[VAL_46]] : vector<8xf32>
// CHECK:                   %[[VAL_69:.*]] = vector.fma %[[VAL_60]], %[[VAL_67]], %[[VAL_47]] : vector<8xf32>
// CHECK:                   %[[VAL_70:.*]] = vector.fma %[[VAL_62]], %[[VAL_67]], %[[VAL_48]] : vector<8xf32>
// CHECK:                   %[[VAL_71:.*]] = vector.load %[[VAL_56]]{{\[}}%[[VAL_3]], %[[VAL_3]], %[[VAL_0]]] : memref<1x1x32xf32, strided<[2048, 64, 1], offset: ?>>, vector<8xf32>
// CHECK:                   %[[VAL_72:.*]] = vector.fma %[[VAL_58]], %[[VAL_71]], %[[VAL_49]] : vector<8xf32>
// CHECK:                   %[[VAL_73:.*]] = vector.fma %[[VAL_60]], %[[VAL_71]], %[[VAL_50]] : vector<8xf32>
// CHECK:                   %[[VAL_74:.*]] = vector.fma %[[VAL_62]], %[[VAL_71]], %[[VAL_51]] : vector<8xf32>
// CHECK:                   %[[VAL_75:.*]] = vector.load %[[VAL_56]]{{\[}}%[[VAL_3]], %[[VAL_3]], %[[VAL_4]]] : memref<1x1x32xf32, strided<[2048, 64, 1], offset: ?>>, vector<8xf32>
// CHECK:                   %[[VAL_76:.*]] = vector.fma %[[VAL_58]], %[[VAL_75]], %[[VAL_52]] : vector<8xf32>
// CHECK:                   %[[VAL_77:.*]] = vector.fma %[[VAL_60]], %[[VAL_75]], %[[VAL_53]] : vector<8xf32>
// CHECK:                   %[[VAL_78:.*]] = vector.fma %[[VAL_62]], %[[VAL_75]], %[[VAL_54]] : vector<8xf32>
// CHECK:                   scf.yield %[[VAL_64]], %[[VAL_65]], %[[VAL_66]], %[[VAL_68]], %[[VAL_69]], %[[VAL_70]], %[[VAL_72]], %[[VAL_73]], %[[VAL_74]], %[[VAL_76]], %[[VAL_77]], %[[VAL_78]] : vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_79:.*]]#0, %[[VAL_79]]#1, %[[VAL_79]]#2, %[[VAL_79]]#3, %[[VAL_79]]#4, %[[VAL_79]]#5, %[[VAL_79]]#6, %[[VAL_79]]#7, %[[VAL_79]]#8, %[[VAL_79]]#9, %[[VAL_79]]#10, %[[VAL_79]]#11 : vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>
// CHECK:               }
// CHECK:               vector.store %[[VAL_80:.*]]#0, %[[VAL_14]]{{\[}}%[[VAL_3]], %[[VAL_3]]] : memref<3x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// CHECK:               vector.store %[[VAL_80]]#1, %[[VAL_14]]{{\[}}%[[VAL_8]], %[[VAL_3]]] : memref<3x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// CHECK:               vector.store %[[VAL_80]]#2, %[[VAL_14]]{{\[}}%[[VAL_2]], %[[VAL_3]]] : memref<3x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// CHECK:               vector.store %[[VAL_80]]#3, %[[VAL_14]]{{\[}}%[[VAL_3]], %[[VAL_1]]] : memref<3x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// CHECK:               vector.store %[[VAL_80]]#4, %[[VAL_14]]{{\[}}%[[VAL_8]], %[[VAL_1]]] : memref<3x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// CHECK:               vector.store %[[VAL_80]]#5, %[[VAL_14]]{{\[}}%[[VAL_2]], %[[VAL_1]]] : memref<3x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// CHECK:               vector.store %[[VAL_80]]#6, %[[VAL_14]]{{\[}}%[[VAL_3]], %[[VAL_0]]] : memref<3x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// CHECK:               vector.store %[[VAL_80]]#7, %[[VAL_14]]{{\[}}%[[VAL_8]], %[[VAL_0]]] : memref<3x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// CHECK:               vector.store %[[VAL_80]]#8, %[[VAL_14]]{{\[}}%[[VAL_2]], %[[VAL_0]]] : memref<3x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// CHECK:               vector.store %[[VAL_80]]#9, %[[VAL_14]]{{\[}}%[[VAL_3]], %[[VAL_4]]] : memref<3x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// CHECK:               vector.store %[[VAL_80]]#10, %[[VAL_14]]{{\[}}%[[VAL_8]], %[[VAL_4]]] : memref<3x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// CHECK:               vector.store %[[VAL_80]]#11, %[[VAL_14]]{{\[}}%[[VAL_2]], %[[VAL_4]]] : memref<3x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

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
