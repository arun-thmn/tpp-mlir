// RUN: tpp-opt %s --vector-contract-to-micro-kernels  --split-input-file  | FileCheck -check-prefix=CHECK %s

module {
  func.func @register_2x3(%arg0: memref<1x2x32xbf16>, %arg1: memref<1x32x48xbf16>, %arg2: memref<2x48xf32>) -> memref<2x48xf32> {
    %0 = ub.poison : f32
    %1 = ub.poison : bf16
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c48 = arith.constant 48 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    scf.for %arg3 = %c0 to %c2 step %c2 {
      scf.for %arg4 = %c0 to %c48 step %c48 {
        %subview = memref.subview %arg2[%arg3, %arg4] [2, 48] [1, 1] : memref<2x48xf32> to memref<2x48xf32, strided<[48, 1], offset: ?>>
        %2 = vector.transfer_read %subview[%c0, %c0], %0 {in_bounds = [true, true]} : memref<2x48xf32, strided<[48, 1], offset: ?>>, vector<2x48xf32>
        %3 = scf.for %arg5 = %c0 to %c1 step %c1 iter_args(%arg6 = %2) -> (vector<2x48xf32>) {
          %4 = scf.for %arg7 = %c0 to %c32 step %c2 iter_args(%arg8 = %arg6) -> (vector<2x48xf32>) {
            %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg7] [1, 2, 2] [1, 1, 1] : memref<1x2x32xbf16> to memref<1x2x2xbf16, strided<[64, 32, 1], offset: ?>>
            %subview_1 = memref.subview %arg1[%arg5, %arg7, %arg4] [1, 2, 48] [1, 1, 1] : memref<1x32x48xbf16> to memref<1x2x48xbf16, strided<[1536, 48, 1], offset: ?>>
            %5 = vector.transfer_read %subview_0[%c0, %c0, %c0], %1 {in_bounds = [true, true, true]} : memref<1x2x2xbf16, strided<[64, 32, 1], offset: ?>>, vector<1x2x2xbf16>
            %6 = vector.transfer_read %subview_1[%c0, %c0, %c0], %1 {in_bounds = [true, true, true]} : memref<1x2x48xbf16, strided<[1536, 48, 1], offset: ?>>, vector<1x2x48xbf16>
            %7 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %5, %6, %arg8 : vector<1x2x2xbf16>, vector<1x2x48xbf16> into vector<2x48xf32>
            scf.yield %7 : vector<2x48xf32>
          }
          scf.yield %4 : vector<2x48xf32>
        }
        vector.transfer_write %3, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<2x48xf32>, memref<2x48xf32, strided<[48, 1], offset: ?>>
      }
    }
    return %arg2 : memref<2x48xf32>
  }
}

// CHECK-LABEL:   func.func @register_2x3(
// CHECK-SAME:      %[[ARG0:.*]]: memref<1x2x32xbf16>,
// CHECK-SAME:      %[[ARG1:.*]]: memref<1x32x48xbf16>,
// CHECK-SAME:      %[[ARG2:.*]]: memref<2x48xf32>) -> memref<2x48xf32> {
// CHECK:           %[[VAL_0:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant dense<0.000000e+00> : vector<16xf32>
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 48 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant -65536 : i32
// CHECK:           %[[VAL_8:.*]] = vector.broadcast %[[VAL_7]] : i32 to vector<16xi32>
// CHECK:           %[[VAL_9:.*]] = memref.alloc() : memref<1xvector<16xi32>>
// CHECK:           memref.store %[[VAL_8]], %[[VAL_9]]{{\[}}%[[VAL_2]]] : memref<1xvector<16xi32>>
// CHECK:           scf.for %[[VAL_10:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_3]] {
// CHECK:             scf.for %[[VAL_11:.*]] = %[[VAL_2]] to %[[VAL_4]] step %[[VAL_4]] {
// CHECK:               %[[VAL_12:.*]] = memref.subview %[[ARG2]]{{\[}}%[[VAL_10]], %[[VAL_11]]] [2, 48] [1, 1] : memref<2x48xf32> to memref<2x48xf32, strided<[48, 1], offset: ?>>
// CHECK:               %[[VAL_13:.*]]:6 = scf.for %[[VAL_14:.*]] = %[[VAL_2]] to %[[VAL_5]] step %[[VAL_5]] iter_args(%[[VAL_15:.*]] = %[[VAL_1]], %[[VAL_16:.*]] = %[[VAL_1]], %[[VAL_17:.*]] = %[[VAL_1]], %[[VAL_18:.*]] = %[[VAL_1]], %[[VAL_19:.*]] = %[[VAL_1]], %[[VAL_20:.*]] = %[[VAL_1]]) -> (vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>) {
// CHECK:                 %[[VAL_21:.*]]:6 = scf.for %[[VAL_22:.*]] = %[[VAL_2]] to %[[VAL_6]] step %[[VAL_3]] iter_args(%[[VAL_23:.*]] = %[[VAL_15]], %[[VAL_24:.*]] = %[[VAL_16]], %[[VAL_25:.*]] = %[[VAL_17]], %[[VAL_26:.*]] = %[[VAL_18]], %[[VAL_27:.*]] = %[[VAL_19]], %[[VAL_28:.*]] = %[[VAL_20]]) -> (vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>) {
// CHECK:                   %[[VAL_29:.*]] = memref.subview %[[ARG0]]{{\[}}%[[VAL_14]], %[[VAL_10]], %[[VAL_22]]] [1, 2, 2] [1, 1, 1] : memref<1x2x32xbf16> to memref<1x2x2xbf16, strided<[64, 32, 1], offset: ?>>
// CHECK:                   %[[VAL_30:.*]] = memref.subview %[[ARG1]]{{\[}}%[[VAL_14]], %[[VAL_22]], %[[VAL_11]]] [1, 2, 48] [1, 1, 1] : memref<1x32x48xbf16> to memref<1x2x48xbf16, strided<[1536, 48, 1], offset: ?>>
// CHECK:                   %[[VAL_31:.*]] = vector.load %[[VAL_29]]{{\[}}%[[VAL_2]], %[[VAL_2]], %[[VAL_2]]] : memref<1x2x2xbf16, strided<[64, 32, 1], offset: ?>>, vector<2xbf16>
// CHECK:                   %[[VAL_32:.*]] = vector.bitcast %[[VAL_31]] : vector<2xbf16> to vector<1xi32>
// CHECK:                   %[[VAL_33:.*]] = vector.broadcast %[[VAL_32]] : vector<1xi32> to vector<16xi32>
// CHECK:                   %[[VAL_34:.*]] = vector.bitcast %[[VAL_33]] : vector<16xi32> to vector<32xbf16>
// CHECK:                   %[[VAL_35:.*]] = vector.load %[[VAL_29]]{{\[}}%[[VAL_2]], %[[VAL_5]], %[[VAL_2]]] : memref<1x2x2xbf16, strided<[64, 32, 1], offset: ?>>, vector<2xbf16>
// CHECK:                   %[[VAL_36:.*]] = vector.bitcast %[[VAL_35]] : vector<2xbf16> to vector<1xi32>
// CHECK:                   %[[VAL_37:.*]] = vector.broadcast %[[VAL_36]] : vector<1xi32> to vector<16xi32>
// CHECK:                   %[[VAL_38:.*]] = vector.bitcast %[[VAL_37]] : vector<16xi32> to vector<32xbf16>
// CHECK:                   %[[VAL_39:.*]] = vector.load %[[VAL_30]]{{\[}}%[[VAL_2]], %[[VAL_2]], %[[VAL_2]]] : memref<1x2x48xbf16, strided<[1536, 48, 1], offset: ?>>, vector<32xbf16>
// CHECK:                   %[[VAL_40:.*]] = vector.load %[[VAL_30]]{{\[}}%[[VAL_2]], %[[VAL_5]], %[[VAL_2]]] : memref<1x2x48xbf16, strided<[1536, 48, 1], offset: ?>>, vector<32xbf16>
// CHECK:                   %[[VAL_41:.*]] = vector.shuffle %[[VAL_39]], %[[VAL_40]] [0, 32, 1, 33, 2, 34, 3, 35, 8, 40, 9, 41, 10, 42, 11, 43, 16, 48, 17, 49, 18, 50, 19, 51, 24, 56, 25, 57, 26, 58, 27, 59] : vector<32xbf16>, vector<32xbf16>
// CHECK:                   %[[VAL_42:.*]] = vector.shuffle %[[VAL_39]], %[[VAL_40]] [4, 36, 5, 37, 6, 38, 7, 39, 12, 44, 13, 45, 14, 46, 15, 47, 20, 52, 21, 53, 22, 54, 23, 55, 28, 60, 29, 61, 30, 62, 31, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK:                   %[[VAL_43:.*]] = x86vector.avx512.dot %[[VAL_23]], %[[VAL_34]], %[[VAL_41]] : vector<32xbf16> -> vector<16xf32>
// CHECK:                   %[[VAL_44:.*]] = x86vector.avx512.dot %[[VAL_26]], %[[VAL_38]], %[[VAL_41]] : vector<32xbf16> -> vector<16xf32>
// CHECK:                   %[[VAL_45:.*]] = x86vector.avx512.dot %[[VAL_24]], %[[VAL_34]], %[[VAL_42]] : vector<32xbf16> -> vector<16xf32>
// CHECK:                   %[[VAL_46:.*]] = x86vector.avx512.dot %[[VAL_27]], %[[VAL_38]], %[[VAL_42]] : vector<32xbf16> -> vector<16xf32>
// CHECK:                   %[[VAL_47:.*]] = vector.load %[[VAL_30]]{{\[}}%[[VAL_2]], %[[VAL_2]], %[[VAL_6]]] : memref<1x2x48xbf16, strided<[1536, 48, 1], offset: ?>>, vector<16xbf16>
// CHECK:                   %[[VAL_48:.*]] = vector.load %[[VAL_30]]{{\[}}%[[VAL_2]], %[[VAL_5]], %[[VAL_6]]] : memref<1x2x48xbf16, strided<[1536, 48, 1], offset: ?>>, vector<16xbf16>
// CHECK:                   %[[VAL_49:.*]] = vector.shuffle %[[VAL_47]], %[[VAL_48]] [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31] : vector<16xbf16>, vector<16xbf16>
// CHECK:                   %[[VAL_50:.*]] = x86vector.avx512.dot %[[VAL_25]], %[[VAL_34]], %[[VAL_49]] : vector<32xbf16> -> vector<16xf32>
// CHECK:                   %[[VAL_51:.*]] = x86vector.avx512.dot %[[VAL_28]], %[[VAL_38]], %[[VAL_49]] : vector<32xbf16> -> vector<16xf32>
// CHECK:                   scf.yield %[[VAL_43]], %[[VAL_45]], %[[VAL_50]], %[[VAL_44]], %[[VAL_46]], %[[VAL_51]] : vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_52:.*]]#0, %[[VAL_52]]#1, %[[VAL_52]]#2, %[[VAL_52]]#3, %[[VAL_52]]#4, %[[VAL_52]]#5 : vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>
// CHECK:               }
// CHECK:               %[[VAL_53:.*]] = vector.shuffle %[[VAL_54:.*]]#0, %[[VAL_54]]#1 [0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
// CHECK:               %[[VAL_55:.*]] = vector.shuffle %[[VAL_54]]#0, %[[VAL_54]]#1 [8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK:               %[[VAL_56:.*]] = vector.load %[[VAL_12]]{{\[}}%[[VAL_2]], %[[VAL_2]]] : memref<2x48xf32, strided<[48, 1], offset: ?>>, vector<16xf32>
// CHECK:               %[[VAL_57:.*]] = arith.addf %[[VAL_53]], %[[VAL_56]] : vector<16xf32>
// CHECK:               %[[VAL_58:.*]] = vector.load %[[VAL_12]]{{\[}}%[[VAL_2]], %[[VAL_0]]] : memref<2x48xf32, strided<[48, 1], offset: ?>>, vector<16xf32>
// CHECK:               %[[VAL_59:.*]] = arith.addf %[[VAL_55]], %[[VAL_58]] : vector<16xf32>
// CHECK:               %[[VAL_60:.*]] = vector.load %[[VAL_12]]{{\[}}%[[VAL_2]], %[[VAL_6]]] : memref<2x48xf32, strided<[48, 1], offset: ?>>, vector<16xf32>
// CHECK:               %[[VAL_61:.*]] = arith.addf %[[VAL_54]]#2, %[[VAL_60]] : vector<16xf32>
// CHECK:               %[[VAL_62:.*]] = vector.shuffle %[[VAL_54]]#3, %[[VAL_54]]#4 [0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
// CHECK:               %[[VAL_63:.*]] = vector.shuffle %[[VAL_54]]#3, %[[VAL_54]]#4 [8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK:               %[[VAL_64:.*]] = vector.load %[[VAL_12]]{{\[}}%[[VAL_5]], %[[VAL_2]]] : memref<2x48xf32, strided<[48, 1], offset: ?>>, vector<16xf32>
// CHECK:               %[[VAL_65:.*]] = arith.addf %[[VAL_62]], %[[VAL_64]] : vector<16xf32>
// CHECK:               %[[VAL_66:.*]] = vector.load %[[VAL_12]]{{\[}}%[[VAL_5]], %[[VAL_0]]] : memref<2x48xf32, strided<[48, 1], offset: ?>>, vector<16xf32>
// CHECK:               %[[VAL_67:.*]] = arith.addf %[[VAL_63]], %[[VAL_66]] : vector<16xf32>
// CHECK:               %[[VAL_68:.*]] = vector.load %[[VAL_12]]{{\[}}%[[VAL_5]], %[[VAL_6]]] : memref<2x48xf32, strided<[48, 1], offset: ?>>, vector<16xf32>
// CHECK:               %[[VAL_69:.*]] = arith.addf %[[VAL_54]]#5, %[[VAL_68]] : vector<16xf32>
// CHECK:               vector.store %[[VAL_57]], %[[VAL_12]]{{\[}}%[[VAL_2]], %[[VAL_2]]] : memref<2x48xf32, strided<[48, 1], offset: ?>>, vector<16xf32>
// CHECK:               vector.store %[[VAL_65]], %[[VAL_12]]{{\[}}%[[VAL_5]], %[[VAL_2]]] : memref<2x48xf32, strided<[48, 1], offset: ?>>, vector<16xf32>
// CHECK:               vector.store %[[VAL_59]], %[[VAL_12]]{{\[}}%[[VAL_2]], %[[VAL_0]]] : memref<2x48xf32, strided<[48, 1], offset: ?>>, vector<16xf32>
// CHECK:               vector.store %[[VAL_67]], %[[VAL_12]]{{\[}}%[[VAL_5]], %[[VAL_0]]] : memref<2x48xf32, strided<[48, 1], offset: ?>>, vector<16xf32>
// CHECK:               vector.store %[[VAL_61]], %[[VAL_12]]{{\[}}%[[VAL_2]], %[[VAL_6]]] : memref<2x48xf32, strided<[48, 1], offset: ?>>, vector<16xf32>
// CHECK:               vector.store %[[VAL_69]], %[[VAL_12]]{{\[}}%[[VAL_5]], %[[VAL_6]]] : memref<2x48xf32, strided<[48, 1], offset: ?>>, vector<16xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return %[[ARG2]] : memref<2x48xf32>
// CHECK:         }

// -----

module {
  func.func @opt_register_9x3_splat(%arg0: memref<1x9x32xbf16>, %arg1: memref<1x32x48xbf16>, %arg2: memref<9x48xbf16>) -> memref<9x48xbf16> {
    %0 = ub.poison : bf16
    %c0 = arith.constant 0 : index
    %c9 = arith.constant 9 : index
    %c48 = arith.constant 48 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    scf.for %arg3 = %c0 to %c9 step %c9 {
      scf.for %arg4 = %c0 to %c48 step %c48 {
        %subview = memref.subview %arg2[%arg3, %arg4] [9, 48] [1, 1] : memref<9x48xbf16> to memref<9x48xbf16, strided<[48, 1], offset: ?>>
        %1 = vector.transfer_read %subview[%c0, %c0], %0 {in_bounds = [true, true]} : memref<9x48xbf16, strided<[48, 1], offset: ?>>, vector<9x48xbf16>
        %2 = scf.for %arg5 = %c0 to %c1 step %c1 iter_args(%arg6 = %1) -> (vector<9x48xbf16>) {
          %3 = scf.for %arg7 = %c0 to %c32 step %c2 iter_args(%arg8 = %arg6) -> (vector<9x48xbf16>) {
            %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg7] [1, 9, 2] [1, 1, 1] : memref<1x9x32xbf16> to memref<1x9x2xbf16, strided<[288, 32, 1], offset: ?>>
            %subview_1 = memref.subview %arg1[%arg5, %arg7, %arg4] [1, 2, 48] [1, 1, 1] : memref<1x32x48xbf16> to memref<1x2x48xbf16, strided<[1536, 48, 1], offset: ?>>
            %4 = vector.transfer_read %subview_0[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} : memref<1x9x2xbf16, strided<[288, 32, 1], offset: ?>>, vector<1x9x2xbf16>
            %5 = vector.transfer_read %subview_1[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} : memref<1x2x48xbf16, strided<[1536, 48, 1], offset: ?>>, vector<1x2x48xbf16>
            %6 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %4, %5, %arg8 : vector<1x9x2xbf16>, vector<1x2x48xbf16> into vector<9x48xbf16>
            scf.yield %6 : vector<9x48xbf16>
          }
          scf.yield %3 : vector<9x48xbf16>
        }
        vector.transfer_write %2, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<9x48xbf16>, memref<9x48xbf16, strided<[48, 1], offset: ?>>
      }
    }
    return %arg2 : memref<9x48xbf16>
  }
}

// CHECK-LABEL:   func.func @opt_register_9x3_splat
// CHECK-COUNT-3: vector.shuffle
// CHECK-COUNT-27: x86vector.avx512.dot

// -----

module {
  func.func @opt_register_6x4_splat(%arg0: memref<1x6x32xbf16>, %arg1: memref<1x32x64xbf16>, %arg2: memref<6x64xf32>) -> memref<6x64xf32> {
    %0 = ub.poison : f32
    %1 = ub.poison : bf16
    %c0 = arith.constant 0 : index
    %c6 = arith.constant 6 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    scf.for %arg3 = %c0 to %c6 step %c6 {
      scf.for %arg4 = %c0 to %c64 step %c64 {
        %subview = memref.subview %arg2[%arg3, %arg4] [6, 64] [1, 1] : memref<6x64xf32> to memref<6x64xf32, strided<[64, 1], offset: ?>>
        %2 = vector.transfer_read %subview[%c0, %c0], %0 {in_bounds = [true, true]} : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<6x64xf32>
        %3 = scf.for %arg5 = %c0 to %c1 step %c1 iter_args(%arg6 = %2) -> (vector<6x64xf32>) {
          %4 = scf.for %arg7 = %c0 to %c32 step %c2 iter_args(%arg8 = %arg6) -> (vector<6x64xf32>) {
            %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg7] [1, 6, 2] [1, 1, 1] : memref<1x6x32xbf16> to memref<1x6x2xbf16, strided<[192, 32, 1], offset: ?>>
            %subview_1 = memref.subview %arg1[%arg5, %arg7, %arg4] [1, 2, 64] [1, 1, 1] : memref<1x32x64xbf16> to memref<1x2x64xbf16, strided<[2048, 64, 1], offset: ?>>
            %5 = vector.transfer_read %subview_0[%c0, %c0, %c0], %1 {in_bounds = [true, true, true]} : memref<1x6x2xbf16, strided<[192, 32, 1], offset: ?>>, vector<1x6x2xbf16>
            %6 = vector.transfer_read %subview_1[%c0, %c0, %c0], %1 {in_bounds = [true, true, true]} : memref<1x2x64xbf16, strided<[2048, 64, 1], offset: ?>>, vector<1x2x64xbf16>
            %7 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %5, %6, %arg8 : vector<1x6x2xbf16>, vector<1x2x64xbf16> into vector<6x64xf32>
            scf.yield %7 : vector<6x64xf32>
          }
          scf.yield %4 : vector<6x64xf32>
        }
        vector.transfer_write %3, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<6x64xf32>, memref<6x64xf32, strided<[64, 1], offset: ?>>
      }
    }
    return %arg2 : memref<6x64xf32>
  }
}

// CHECK-LABEL:   func.func @opt_register_6x4_splat
// CHECK-COUNT-3: vector.shuffle
// CHECK-COUNT-24: x86vector.avx512.dot
