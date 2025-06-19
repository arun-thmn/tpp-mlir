// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=6,64,1" --loop-invariant-code-motion --vectorization-pass --hoist-vector-transfer --vector-contract-to-ukernels  --split-input-file  | FileCheck -check-prefix=CHECK %s
// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=6,8,1" --loop-invariant-code-motion --vectorization-pass --hoist-vector-transfer --vector-contract-to-ukernels  --split-input-file  | FileCheck -check-prefix=CHECK1 %s
// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=6,64,4" --loop-invariant-code-motion --vectorization-pass --hoist-vector-transfer --vector-contract-to-ukernels  --split-input-file  | FileCheck -check-prefix=CHECK2 %s

module {
 func.func @optimal_register_allocation(%arg0: memref<32x24x32xf32>, %arg1: memref<32x32x64xf32>, %arg2: memref<24x64xf32>) -> memref<24x64xf32> {
     linalg.batch_reduce_matmul ins(%arg0, %arg1 : memref<32x24x32xf32>, memref<32x32x64xf32>) outs(%arg2 : memref<24x64xf32>)
   return %arg2 : memref<24x64xf32>
 }
}

// CHECK-LABEL:   func.func @optimal_register_allocation(
// CHECK-SAME:                     %[[ARG0:.*]]: memref<32x24x32xf32>,
// CHECK-SAME:                     %[[ARG1:.*]]: memref<32x32x64xf32>,
// CHECK-SAME:                     %[[ARG2:.*]]: memref<24x64xf32>) -> memref<24x64xf32> {
// CHECK:           %[[VAL_0:.*]] = arith.constant 48 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 5 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 24 : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 6 : index
// CHECK:           %[[VAL_9:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_10:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_11:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_12:.*]] = arith.constant -65536 : i32
// CHECK:           %[[VAL_13:.*]] = vector.broadcast %[[VAL_12]] : i32 to vector<16xi32>
// CHECK:           %[[VAL_14:.*]] = memref.alloc() : memref<1xvector<16xi32>>
// CHECK:           memref.store %[[VAL_13]], %[[VAL_14]]{{\[}}%[[VAL_6]]] : memref<1xvector<16xi32>>
// CHECK:           scf.for %[[VAL_15:.*]] = %[[VAL_6]] to %[[VAL_7]] step %[[VAL_8]] {
// CHECK:             scf.for %[[VAL_16:.*]] = %[[VAL_6]] to %[[VAL_9]] step %[[VAL_9]] {
// CHECK:               %[[VAL_17:.*]] = memref.subview %[[ARG2]]{{\[}}%[[VAL_15]], %[[VAL_16]]] [6, 64] [1, 1] : memref<24x64xf32> to memref<6x64xf32, strided<[64, 1], offset: ?>>
// CHECK:               %[[VAL_18:.*]] = vector.load %[[VAL_17]]{{\[}}%[[VAL_6]], %[[VAL_6]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               %[[VAL_19:.*]] = vector.load %[[VAL_17]]{{\[}}%[[VAL_11]], %[[VAL_6]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               %[[VAL_20:.*]] = vector.load %[[VAL_17]]{{\[}}%[[VAL_5]], %[[VAL_6]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               %[[VAL_21:.*]] = vector.load %[[VAL_17]]{{\[}}%[[VAL_4]], %[[VAL_6]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               %[[VAL_22:.*]] = vector.load %[[VAL_17]]{{\[}}%[[VAL_3]], %[[VAL_6]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               %[[VAL_23:.*]] = vector.load %[[VAL_17]]{{\[}}%[[VAL_2]], %[[VAL_6]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               %[[VAL_24:.*]] = vector.load %[[VAL_17]]{{\[}}%[[VAL_6]], %[[VAL_1]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               %[[VAL_25:.*]] = vector.load %[[VAL_17]]{{\[}}%[[VAL_11]], %[[VAL_1]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               %[[VAL_26:.*]] = vector.load %[[VAL_17]]{{\[}}%[[VAL_5]], %[[VAL_1]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               %[[VAL_27:.*]] = vector.load %[[VAL_17]]{{\[}}%[[VAL_4]], %[[VAL_1]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               %[[VAL_28:.*]] = vector.load %[[VAL_17]]{{\[}}%[[VAL_3]], %[[VAL_1]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               %[[VAL_29:.*]] = vector.load %[[VAL_17]]{{\[}}%[[VAL_2]], %[[VAL_1]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               %[[VAL_30:.*]] = vector.load %[[VAL_17]]{{\[}}%[[VAL_6]], %[[VAL_10]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               %[[VAL_31:.*]] = vector.load %[[VAL_17]]{{\[}}%[[VAL_11]], %[[VAL_10]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               %[[VAL_32:.*]] = vector.load %[[VAL_17]]{{\[}}%[[VAL_5]], %[[VAL_10]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               %[[VAL_33:.*]] = vector.load %[[VAL_17]]{{\[}}%[[VAL_4]], %[[VAL_10]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               %[[VAL_34:.*]] = vector.load %[[VAL_17]]{{\[}}%[[VAL_3]], %[[VAL_10]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               %[[VAL_35:.*]] = vector.load %[[VAL_17]]{{\[}}%[[VAL_2]], %[[VAL_10]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               %[[VAL_36:.*]] = vector.load %[[VAL_17]]{{\[}}%[[VAL_6]], %[[VAL_0]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               %[[VAL_37:.*]] = vector.load %[[VAL_17]]{{\[}}%[[VAL_11]], %[[VAL_0]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               %[[VAL_38:.*]] = vector.load %[[VAL_17]]{{\[}}%[[VAL_5]], %[[VAL_0]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               %[[VAL_39:.*]] = vector.load %[[VAL_17]]{{\[}}%[[VAL_4]], %[[VAL_0]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               %[[VAL_40:.*]] = vector.load %[[VAL_17]]{{\[}}%[[VAL_3]], %[[VAL_0]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               %[[VAL_41:.*]] = vector.load %[[VAL_17]]{{\[}}%[[VAL_2]], %[[VAL_0]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               %[[VAL_42:.*]]:24 = scf.for %[[VAL_43:.*]] = %[[VAL_6]] to %[[VAL_10]] step %[[VAL_11]] iter_args(%[[VAL_44:.*]] = %[[VAL_18]], %[[VAL_45:.*]] = %[[VAL_19]], %[[VAL_46:.*]] = %[[VAL_20]], %[[VAL_47:.*]] = %[[VAL_21]], %[[VAL_48:.*]] = %[[VAL_22]], %[[VAL_49:.*]] = %[[VAL_23]], %[[VAL_50:.*]] = %[[VAL_24]], %[[VAL_51:.*]] = %[[VAL_25]], %[[VAL_52:.*]] = %[[VAL_26]], %[[VAL_53:.*]] = %[[VAL_27]], %[[VAL_54:.*]] = %[[VAL_28]], %[[VAL_55:.*]] = %[[VAL_29]], %[[VAL_56:.*]] = %[[VAL_30]], %[[VAL_57:.*]] = %[[VAL_31]], %[[VAL_58:.*]] = %[[VAL_32]], %[[VAL_59:.*]] = %[[VAL_33]], %[[VAL_60:.*]] = %[[VAL_34]], %[[VAL_61:.*]] = %[[VAL_35]], %[[VAL_62:.*]] = %[[VAL_36]], %[[VAL_63:.*]] = %[[VAL_37]], %[[VAL_64:.*]] = %[[VAL_38]], %[[VAL_65:.*]] = %[[VAL_39]], %[[VAL_66:.*]] = %[[VAL_40]], %[[VAL_67:.*]] = %[[VAL_41]]) -> (vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>) {
// CHECK:                 %[[VAL_68:.*]]:24 = scf.for %[[VAL_69:.*]] = %[[VAL_6]] to %[[VAL_10]] step %[[VAL_11]] iter_args(%[[VAL_70:.*]] = %[[VAL_44]], %[[VAL_71:.*]] = %[[VAL_45]], %[[VAL_72:.*]] = %[[VAL_46]], %[[VAL_73:.*]] = %[[VAL_47]], %[[VAL_74:.*]] = %[[VAL_48]], %[[VAL_75:.*]] = %[[VAL_49]], %[[VAL_76:.*]] = %[[VAL_50]], %[[VAL_77:.*]] = %[[VAL_51]], %[[VAL_78:.*]] = %[[VAL_52]], %[[VAL_79:.*]] = %[[VAL_53]], %[[VAL_80:.*]] = %[[VAL_54]], %[[VAL_81:.*]] = %[[VAL_55]], %[[VAL_82:.*]] = %[[VAL_56]], %[[VAL_83:.*]] = %[[VAL_57]], %[[VAL_84:.*]] = %[[VAL_58]], %[[VAL_85:.*]] = %[[VAL_59]], %[[VAL_86:.*]] = %[[VAL_60]], %[[VAL_87:.*]] = %[[VAL_61]], %[[VAL_88:.*]] = %[[VAL_62]], %[[VAL_89:.*]] = %[[VAL_63]], %[[VAL_90:.*]] = %[[VAL_64]], %[[VAL_91:.*]] = %[[VAL_65]], %[[VAL_92:.*]] = %[[VAL_66]], %[[VAL_93:.*]] = %[[VAL_67]]) -> (vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>) {
// CHECK:                   %[[VAL_94:.*]] = memref.subview %[[ARG0]]{{\[}}%[[VAL_43]], %[[VAL_15]], %[[VAL_69]]] [1, 6, 1] [1, 1, 1] : memref<32x24x32xf32> to memref<1x6x1xf32, strided<[768, 32, 1], offset: ?>>
// CHECK:                   %[[VAL_95:.*]] = memref.subview %[[ARG1]]{{\[}}%[[VAL_43]], %[[VAL_69]], %[[VAL_16]]] [1, 1, 64] [1, 1, 1] : memref<32x32x64xf32> to memref<1x1x64xf32, strided<[2048, 64, 1], offset: ?>>
// CHECK:                   %[[VAL_96:.*]] = vector.load %[[VAL_95]]{{\[}}%[[VAL_6]], %[[VAL_6]], %[[VAL_6]]] : memref<1x1x64xf32, strided<[2048, 64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   %[[VAL_97:.*]] = vector.load %[[VAL_95]]{{\[}}%[[VAL_6]], %[[VAL_6]], %[[VAL_1]]] : memref<1x1x64xf32, strided<[2048, 64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   %[[VAL_98:.*]] = vector.load %[[VAL_95]]{{\[}}%[[VAL_6]], %[[VAL_6]], %[[VAL_10]]] : memref<1x1x64xf32, strided<[2048, 64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   %[[VAL_99:.*]] = vector.load %[[VAL_95]]{{\[}}%[[VAL_6]], %[[VAL_6]], %[[VAL_0]]] : memref<1x1x64xf32, strided<[2048, 64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   %[[VAL_100:.*]] = vector.load %[[VAL_94]]{{\[}}%[[VAL_6]], %[[VAL_6]], %[[VAL_6]]] : memref<1x6x1xf32, strided<[768, 32, 1], offset: ?>>, vector<1xf32>
// CHECK:                   %[[VAL_101:.*]] = vector.broadcast %[[VAL_100]] : vector<1xf32> to vector<16xf32>
// CHECK:                   %[[VAL_102:.*]] = vector.fma %[[VAL_101]], %[[VAL_96]], %[[VAL_70]] : vector<16xf32>
// CHECK:                   %[[VAL_103:.*]] = vector.fma %[[VAL_101]], %[[VAL_97]], %[[VAL_76]] : vector<16xf32>
// CHECK:                   %[[VAL_104:.*]] = vector.fma %[[VAL_101]], %[[VAL_98]], %[[VAL_82]] : vector<16xf32>
// CHECK:                   %[[VAL_105:.*]] = vector.fma %[[VAL_101]], %[[VAL_99]], %[[VAL_88]] : vector<16xf32>
// CHECK:                   %[[VAL_106:.*]] = vector.load %[[VAL_94]]{{\[}}%[[VAL_6]], %[[VAL_11]], %[[VAL_6]]] : memref<1x6x1xf32, strided<[768, 32, 1], offset: ?>>, vector<1xf32>
// CHECK:                   %[[VAL_107:.*]] = vector.broadcast %[[VAL_106]] : vector<1xf32> to vector<16xf32>
// CHECK:                   %[[VAL_108:.*]] = vector.fma %[[VAL_107]], %[[VAL_96]], %[[VAL_71]] : vector<16xf32>
// CHECK:                   %[[VAL_109:.*]] = vector.fma %[[VAL_107]], %[[VAL_97]], %[[VAL_77]] : vector<16xf32>
// CHECK:                   %[[VAL_110:.*]] = vector.fma %[[VAL_107]], %[[VAL_98]], %[[VAL_83]] : vector<16xf32>
// CHECK:                   %[[VAL_111:.*]] = vector.fma %[[VAL_107]], %[[VAL_99]], %[[VAL_89]] : vector<16xf32>
// CHECK:                   %[[VAL_112:.*]] = vector.load %[[VAL_94]]{{\[}}%[[VAL_6]], %[[VAL_5]], %[[VAL_6]]] : memref<1x6x1xf32, strided<[768, 32, 1], offset: ?>>, vector<1xf32>
// CHECK:                   %[[VAL_113:.*]] = vector.broadcast %[[VAL_112]] : vector<1xf32> to vector<16xf32>
// CHECK:                   %[[VAL_114:.*]] = vector.fma %[[VAL_113]], %[[VAL_96]], %[[VAL_72]] : vector<16xf32>
// CHECK:                   %[[VAL_115:.*]] = vector.fma %[[VAL_113]], %[[VAL_97]], %[[VAL_78]] : vector<16xf32>
// CHECK:                   %[[VAL_116:.*]] = vector.fma %[[VAL_113]], %[[VAL_98]], %[[VAL_84]] : vector<16xf32>
// CHECK:                   %[[VAL_117:.*]] = vector.fma %[[VAL_113]], %[[VAL_99]], %[[VAL_90]] : vector<16xf32>
// CHECK:                   %[[VAL_118:.*]] = vector.load %[[VAL_94]]{{\[}}%[[VAL_6]], %[[VAL_4]], %[[VAL_6]]] : memref<1x6x1xf32, strided<[768, 32, 1], offset: ?>>, vector<1xf32>
// CHECK:                   %[[VAL_119:.*]] = vector.broadcast %[[VAL_118]] : vector<1xf32> to vector<16xf32>
// CHECK:                   %[[VAL_120:.*]] = vector.fma %[[VAL_119]], %[[VAL_96]], %[[VAL_73]] : vector<16xf32>
// CHECK:                   %[[VAL_121:.*]] = vector.fma %[[VAL_119]], %[[VAL_97]], %[[VAL_79]] : vector<16xf32>
// CHECK:                   %[[VAL_122:.*]] = vector.fma %[[VAL_119]], %[[VAL_98]], %[[VAL_85]] : vector<16xf32>
// CHECK:                   %[[VAL_123:.*]] = vector.fma %[[VAL_119]], %[[VAL_99]], %[[VAL_91]] : vector<16xf32>
// CHECK:                   %[[VAL_124:.*]] = vector.load %[[VAL_94]]{{\[}}%[[VAL_6]], %[[VAL_3]], %[[VAL_6]]] : memref<1x6x1xf32, strided<[768, 32, 1], offset: ?>>, vector<1xf32>
// CHECK:                   %[[VAL_125:.*]] = vector.broadcast %[[VAL_124]] : vector<1xf32> to vector<16xf32>
// CHECK:                   %[[VAL_126:.*]] = vector.fma %[[VAL_125]], %[[VAL_96]], %[[VAL_74]] : vector<16xf32>
// CHECK:                   %[[VAL_127:.*]] = vector.fma %[[VAL_125]], %[[VAL_97]], %[[VAL_80]] : vector<16xf32>
// CHECK:                   %[[VAL_128:.*]] = vector.fma %[[VAL_125]], %[[VAL_98]], %[[VAL_86]] : vector<16xf32>
// CHECK:                   %[[VAL_129:.*]] = vector.fma %[[VAL_125]], %[[VAL_99]], %[[VAL_92]] : vector<16xf32>
// CHECK:                   %[[VAL_130:.*]] = vector.load %[[VAL_94]]{{\[}}%[[VAL_6]], %[[VAL_2]], %[[VAL_6]]] : memref<1x6x1xf32, strided<[768, 32, 1], offset: ?>>, vector<1xf32>
// CHECK:                   %[[VAL_131:.*]] = vector.broadcast %[[VAL_130]] : vector<1xf32> to vector<16xf32>
// CHECK:                   %[[VAL_132:.*]] = vector.fma %[[VAL_131]], %[[VAL_96]], %[[VAL_75]] : vector<16xf32>
// CHECK:                   %[[VAL_133:.*]] = vector.fma %[[VAL_131]], %[[VAL_97]], %[[VAL_81]] : vector<16xf32>
// CHECK:                   %[[VAL_134:.*]] = vector.fma %[[VAL_131]], %[[VAL_98]], %[[VAL_87]] : vector<16xf32>
// CHECK:                   %[[VAL_135:.*]] = vector.fma %[[VAL_131]], %[[VAL_99]], %[[VAL_93]] : vector<16xf32>
// CHECK:                   scf.yield %[[VAL_102]], %[[VAL_108]], %[[VAL_114]], %[[VAL_120]], %[[VAL_126]], %[[VAL_132]], %[[VAL_103]], %[[VAL_109]], %[[VAL_115]], %[[VAL_121]], %[[VAL_127]], %[[VAL_133]], %[[VAL_104]], %[[VAL_110]], %[[VAL_116]], %[[VAL_122]], %[[VAL_128]], %[[VAL_134]], %[[VAL_105]], %[[VAL_111]], %[[VAL_117]], %[[VAL_123]], %[[VAL_129]], %[[VAL_135]] : vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_136:.*]]#0, %[[VAL_136]]#1, %[[VAL_136]]#2, %[[VAL_136]]#3, %[[VAL_136]]#4, %[[VAL_136]]#5, %[[VAL_136]]#6, %[[VAL_136]]#7, %[[VAL_136]]#8, %[[VAL_136]]#9, %[[VAL_136]]#10, %[[VAL_136]]#11, %[[VAL_136]]#12, %[[VAL_136]]#13, %[[VAL_136]]#14, %[[VAL_136]]#15, %[[VAL_136]]#16, %[[VAL_136]]#17, %[[VAL_136]]#18, %[[VAL_136]]#19, %[[VAL_136]]#20, %[[VAL_136]]#21, %[[VAL_136]]#22, %[[VAL_136]]#23 : vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>
// CHECK:               }
// CHECK:               vector.store %[[VAL_137:.*]]#0, %[[VAL_17]]{{\[}}%[[VAL_6]], %[[VAL_6]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               vector.store %[[VAL_137]]#1, %[[VAL_17]]{{\[}}%[[VAL_11]], %[[VAL_6]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               vector.store %[[VAL_137]]#2, %[[VAL_17]]{{\[}}%[[VAL_5]], %[[VAL_6]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               vector.store %[[VAL_137]]#3, %[[VAL_17]]{{\[}}%[[VAL_4]], %[[VAL_6]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               vector.store %[[VAL_137]]#4, %[[VAL_17]]{{\[}}%[[VAL_3]], %[[VAL_6]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               vector.store %[[VAL_137]]#5, %[[VAL_17]]{{\[}}%[[VAL_2]], %[[VAL_6]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               vector.store %[[VAL_137]]#6, %[[VAL_17]]{{\[}}%[[VAL_6]], %[[VAL_1]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               vector.store %[[VAL_137]]#7, %[[VAL_17]]{{\[}}%[[VAL_11]], %[[VAL_1]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               vector.store %[[VAL_137]]#8, %[[VAL_17]]{{\[}}%[[VAL_5]], %[[VAL_1]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               vector.store %[[VAL_137]]#9, %[[VAL_17]]{{\[}}%[[VAL_4]], %[[VAL_1]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               vector.store %[[VAL_137]]#10, %[[VAL_17]]{{\[}}%[[VAL_3]], %[[VAL_1]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               vector.store %[[VAL_137]]#11, %[[VAL_17]]{{\[}}%[[VAL_2]], %[[VAL_1]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               vector.store %[[VAL_137]]#12, %[[VAL_17]]{{\[}}%[[VAL_6]], %[[VAL_10]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               vector.store %[[VAL_137]]#13, %[[VAL_17]]{{\[}}%[[VAL_11]], %[[VAL_10]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               vector.store %[[VAL_137]]#14, %[[VAL_17]]{{\[}}%[[VAL_5]], %[[VAL_10]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               vector.store %[[VAL_137]]#15, %[[VAL_17]]{{\[}}%[[VAL_4]], %[[VAL_10]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               vector.store %[[VAL_137]]#16, %[[VAL_17]]{{\[}}%[[VAL_3]], %[[VAL_10]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               vector.store %[[VAL_137]]#17, %[[VAL_17]]{{\[}}%[[VAL_2]], %[[VAL_10]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               vector.store %[[VAL_137]]#18, %[[VAL_17]]{{\[}}%[[VAL_6]], %[[VAL_0]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               vector.store %[[VAL_137]]#19, %[[VAL_17]]{{\[}}%[[VAL_11]], %[[VAL_0]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               vector.store %[[VAL_137]]#20, %[[VAL_17]]{{\[}}%[[VAL_5]], %[[VAL_0]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               vector.store %[[VAL_137]]#21, %[[VAL_17]]{{\[}}%[[VAL_4]], %[[VAL_0]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               vector.store %[[VAL_137]]#22, %[[VAL_17]]{{\[}}%[[VAL_3]], %[[VAL_0]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:               vector.store %[[VAL_137]]#23, %[[VAL_17]]{{\[}}%[[VAL_2]], %[[VAL_0]]] : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return %[[ARG2]] : memref<24x64xf32>
// CHECK:         }

// CHECK1-LABEL: @optimal_register_allocation
// CHECK1-NOT: vector.fma
// CHECK1: vector.contract

// CHECK2-LABEL: @optimal_register_allocation
// CHECK2-NOT: vector.fma
// CHECK2: vector.contract


// -----
