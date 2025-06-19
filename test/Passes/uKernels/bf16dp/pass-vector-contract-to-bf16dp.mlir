// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=6,64,2" --loop-invariant-code-motion --vectorization-pass --hoist-vector-transfer --vector-contract-to-ukernels  --split-input-file  | FileCheck -check-prefix=CHECK %s

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

// CHECK-LABEL:   memref.global "private" constant @__constant_2x16x128x2xbf16 : memref<2x16x128x2xbf16> = dense<1.000000e+00> {alignment = 64 : i64}
// CHECK-LABEL:   func.func @optimal_register_allocation(
// CHECK-SAME:                     %[[ARG0:.*]]: memref<2x24x16x2xbf16>) -> memref<24x128xbf16> {
// CHECK:           %[[VAL_0:.*]] = arith.constant 48 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 5 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant -65536 : i32
// CHECK:           %[[VAL_6:.*]] = arith.constant 16 : i32
// CHECK:           %[[VAL_7:.*]] = arith.constant dense<0.000000e+00> : vector<24x128xbf16>
// CHECK:           %[[VAL_8:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_9:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_10:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_11:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_12:.*]] = arith.constant 128 : index
// CHECK:           %[[VAL_13:.*]] = arith.constant 6 : index
// CHECK:           %[[VAL_14:.*]] = arith.constant 24 : index
// CHECK:           %[[VAL_15:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_16:.*]] = memref.get_global @__constant_2x16x128x2xbf16 : memref<2x16x128x2xbf16>
// CHECK:           %[[VAL_17:.*]] = memref.alloc() {alignment = 64 : i64} : memref<24x128xbf16>
// CHECK:           vector.transfer_write %[[VAL_7]], %[[VAL_17]]{{\[}}%[[VAL_15]], %[[VAL_15]]] {in_bounds = [true, true]} : vector<24x128xbf16>, memref<24x128xbf16>
// CHECK:           %[[VAL_18:.*]] = vector.broadcast %[[VAL_5]] : i32 to vector<16xi32>
// CHECK:           %[[VAL_19:.*]] = memref.alloc() : memref<1xvector<16xi32>>
// CHECK:           memref.store %[[VAL_18]], %[[VAL_19]]{{\[}}%[[VAL_15]]] : memref<1xvector<16xi32>>
// CHECK:           scf.for %[[VAL_20:.*]] = %[[VAL_15]] to %[[VAL_14]] step %[[VAL_13]] {
// CHECK:             scf.for %[[VAL_21:.*]] = %[[VAL_15]] to %[[VAL_12]] step %[[VAL_11]] {
// CHECK:               %[[VAL_22:.*]] = memref.subview %[[VAL_17]]{{\[}}%[[VAL_20]], %[[VAL_21]]] [6, 64] [1, 1] : memref<24x128xbf16> to memref<6x64xbf16, strided<[128, 1], offset: ?>>
// CHECK:               %[[VAL_23:.*]] = vector.load %[[VAL_22]]{{\[}}%[[VAL_15]], %[[VAL_15]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_24:.*]] = vector.bitcast %[[VAL_23]] : vector<16xbf16> to vector<16xi16>
// CHECK:               %[[VAL_25:.*]] = arith.extui %[[VAL_24]] : vector<16xi16> to vector<16xi32>
// CHECK:               %[[VAL_26:.*]] = vector.broadcast %[[VAL_6]] : i32 to vector<16xi32>
// CHECK:               %[[VAL_27:.*]] = arith.shli %[[VAL_25]], %[[VAL_26]] : vector<16xi32>
// CHECK:               %[[VAL_28:.*]] = vector.bitcast %[[VAL_27]] : vector<16xi32> to vector<16xf32>
// CHECK:               %[[VAL_29:.*]] = vector.load %[[VAL_22]]{{\[}}%[[VAL_9]], %[[VAL_15]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_30:.*]] = vector.bitcast %[[VAL_29]] : vector<16xbf16> to vector<16xi16>
// CHECK:               %[[VAL_31:.*]] = arith.extui %[[VAL_30]] : vector<16xi16> to vector<16xi32>
// CHECK:               %[[VAL_32:.*]] = vector.broadcast %[[VAL_6]] : i32 to vector<16xi32>
// CHECK:               %[[VAL_33:.*]] = arith.shli %[[VAL_31]], %[[VAL_32]] : vector<16xi32>
// CHECK:               %[[VAL_34:.*]] = vector.bitcast %[[VAL_33]] : vector<16xi32> to vector<16xf32>
// CHECK:               %[[VAL_35:.*]] = vector.load %[[VAL_22]]{{\[}}%[[VAL_10]], %[[VAL_15]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_36:.*]] = vector.bitcast %[[VAL_35]] : vector<16xbf16> to vector<16xi16>
// CHECK:               %[[VAL_37:.*]] = arith.extui %[[VAL_36]] : vector<16xi16> to vector<16xi32>
// CHECK:               %[[VAL_38:.*]] = vector.broadcast %[[VAL_6]] : i32 to vector<16xi32>
// CHECK:               %[[VAL_39:.*]] = arith.shli %[[VAL_37]], %[[VAL_38]] : vector<16xi32>
// CHECK:               %[[VAL_40:.*]] = vector.bitcast %[[VAL_39]] : vector<16xi32> to vector<16xf32>
// CHECK:               %[[VAL_41:.*]] = vector.load %[[VAL_22]]{{\[}}%[[VAL_4]], %[[VAL_15]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_42:.*]] = vector.bitcast %[[VAL_41]] : vector<16xbf16> to vector<16xi16>
// CHECK:               %[[VAL_43:.*]] = arith.extui %[[VAL_42]] : vector<16xi16> to vector<16xi32>
// CHECK:               %[[VAL_44:.*]] = vector.broadcast %[[VAL_6]] : i32 to vector<16xi32>
// CHECK:               %[[VAL_45:.*]] = arith.shli %[[VAL_43]], %[[VAL_44]] : vector<16xi32>
// CHECK:               %[[VAL_46:.*]] = vector.bitcast %[[VAL_45]] : vector<16xi32> to vector<16xf32>
// CHECK:               %[[VAL_47:.*]] = vector.load %[[VAL_22]]{{\[}}%[[VAL_3]], %[[VAL_15]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_48:.*]] = vector.bitcast %[[VAL_47]] : vector<16xbf16> to vector<16xi16>
// CHECK:               %[[VAL_49:.*]] = arith.extui %[[VAL_48]] : vector<16xi16> to vector<16xi32>
// CHECK:               %[[VAL_50:.*]] = vector.broadcast %[[VAL_6]] : i32 to vector<16xi32>
// CHECK:               %[[VAL_51:.*]] = arith.shli %[[VAL_49]], %[[VAL_50]] : vector<16xi32>
// CHECK:               %[[VAL_52:.*]] = vector.bitcast %[[VAL_51]] : vector<16xi32> to vector<16xf32>
// CHECK:               %[[VAL_53:.*]] = vector.load %[[VAL_22]]{{\[}}%[[VAL_2]], %[[VAL_15]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_54:.*]] = vector.bitcast %[[VAL_53]] : vector<16xbf16> to vector<16xi16>
// CHECK:               %[[VAL_55:.*]] = arith.extui %[[VAL_54]] : vector<16xi16> to vector<16xi32>
// CHECK:               %[[VAL_56:.*]] = vector.broadcast %[[VAL_6]] : i32 to vector<16xi32>
// CHECK:               %[[VAL_57:.*]] = arith.shli %[[VAL_55]], %[[VAL_56]] : vector<16xi32>
// CHECK:               %[[VAL_58:.*]] = vector.bitcast %[[VAL_57]] : vector<16xi32> to vector<16xf32>
// CHECK:               %[[VAL_59:.*]] = vector.load %[[VAL_22]]{{\[}}%[[VAL_15]], %[[VAL_8]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_60:.*]] = vector.bitcast %[[VAL_59]] : vector<16xbf16> to vector<16xi16>
// CHECK:               %[[VAL_61:.*]] = arith.extui %[[VAL_60]] : vector<16xi16> to vector<16xi32>
// CHECK:               %[[VAL_62:.*]] = vector.broadcast %[[VAL_6]] : i32 to vector<16xi32>
// CHECK:               %[[VAL_63:.*]] = arith.shli %[[VAL_61]], %[[VAL_62]] : vector<16xi32>
// CHECK:               %[[VAL_64:.*]] = vector.bitcast %[[VAL_63]] : vector<16xi32> to vector<16xf32>
// CHECK:               %[[VAL_65:.*]] = vector.load %[[VAL_22]]{{\[}}%[[VAL_9]], %[[VAL_8]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_66:.*]] = vector.bitcast %[[VAL_65]] : vector<16xbf16> to vector<16xi16>
// CHECK:               %[[VAL_67:.*]] = arith.extui %[[VAL_66]] : vector<16xi16> to vector<16xi32>
// CHECK:               %[[VAL_68:.*]] = vector.broadcast %[[VAL_6]] : i32 to vector<16xi32>
// CHECK:               %[[VAL_69:.*]] = arith.shli %[[VAL_67]], %[[VAL_68]] : vector<16xi32>
// CHECK:               %[[VAL_70:.*]] = vector.bitcast %[[VAL_69]] : vector<16xi32> to vector<16xf32>
// CHECK:               %[[VAL_71:.*]] = vector.load %[[VAL_22]]{{\[}}%[[VAL_10]], %[[VAL_8]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_72:.*]] = vector.bitcast %[[VAL_71]] : vector<16xbf16> to vector<16xi16>
// CHECK:               %[[VAL_73:.*]] = arith.extui %[[VAL_72]] : vector<16xi16> to vector<16xi32>
// CHECK:               %[[VAL_74:.*]] = vector.broadcast %[[VAL_6]] : i32 to vector<16xi32>
// CHECK:               %[[VAL_75:.*]] = arith.shli %[[VAL_73]], %[[VAL_74]] : vector<16xi32>
// CHECK:               %[[VAL_76:.*]] = vector.bitcast %[[VAL_75]] : vector<16xi32> to vector<16xf32>
// CHECK:               %[[VAL_77:.*]] = vector.load %[[VAL_22]]{{\[}}%[[VAL_4]], %[[VAL_8]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_78:.*]] = vector.bitcast %[[VAL_77]] : vector<16xbf16> to vector<16xi16>
// CHECK:               %[[VAL_79:.*]] = arith.extui %[[VAL_78]] : vector<16xi16> to vector<16xi32>
// CHECK:               %[[VAL_80:.*]] = vector.broadcast %[[VAL_6]] : i32 to vector<16xi32>
// CHECK:               %[[VAL_81:.*]] = arith.shli %[[VAL_79]], %[[VAL_80]] : vector<16xi32>
// CHECK:               %[[VAL_82:.*]] = vector.bitcast %[[VAL_81]] : vector<16xi32> to vector<16xf32>
// CHECK:               %[[VAL_83:.*]] = vector.load %[[VAL_22]]{{\[}}%[[VAL_3]], %[[VAL_8]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_84:.*]] = vector.bitcast %[[VAL_83]] : vector<16xbf16> to vector<16xi16>
// CHECK:               %[[VAL_85:.*]] = arith.extui %[[VAL_84]] : vector<16xi16> to vector<16xi32>
// CHECK:               %[[VAL_86:.*]] = vector.broadcast %[[VAL_6]] : i32 to vector<16xi32>
// CHECK:               %[[VAL_87:.*]] = arith.shli %[[VAL_85]], %[[VAL_86]] : vector<16xi32>
// CHECK:               %[[VAL_88:.*]] = vector.bitcast %[[VAL_87]] : vector<16xi32> to vector<16xf32>
// CHECK:               %[[VAL_89:.*]] = vector.load %[[VAL_22]]{{\[}}%[[VAL_2]], %[[VAL_8]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_90:.*]] = vector.bitcast %[[VAL_89]] : vector<16xbf16> to vector<16xi16>
// CHECK:               %[[VAL_91:.*]] = arith.extui %[[VAL_90]] : vector<16xi16> to vector<16xi32>
// CHECK:               %[[VAL_92:.*]] = vector.broadcast %[[VAL_6]] : i32 to vector<16xi32>
// CHECK:               %[[VAL_93:.*]] = arith.shli %[[VAL_91]], %[[VAL_92]] : vector<16xi32>
// CHECK:               %[[VAL_94:.*]] = vector.bitcast %[[VAL_93]] : vector<16xi32> to vector<16xf32>
// CHECK:               %[[VAL_95:.*]] = vector.load %[[VAL_22]]{{\[}}%[[VAL_15]], %[[VAL_1]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_96:.*]] = vector.bitcast %[[VAL_95]] : vector<16xbf16> to vector<16xi16>
// CHECK:               %[[VAL_97:.*]] = arith.extui %[[VAL_96]] : vector<16xi16> to vector<16xi32>
// CHECK:               %[[VAL_98:.*]] = vector.broadcast %[[VAL_6]] : i32 to vector<16xi32>
// CHECK:               %[[VAL_99:.*]] = arith.shli %[[VAL_97]], %[[VAL_98]] : vector<16xi32>
// CHECK:               %[[VAL_100:.*]] = vector.bitcast %[[VAL_99]] : vector<16xi32> to vector<16xf32>
// CHECK:               %[[VAL_101:.*]] = vector.load %[[VAL_22]]{{\[}}%[[VAL_9]], %[[VAL_1]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_102:.*]] = vector.bitcast %[[VAL_101]] : vector<16xbf16> to vector<16xi16>
// CHECK:               %[[VAL_103:.*]] = arith.extui %[[VAL_102]] : vector<16xi16> to vector<16xi32>
// CHECK:               %[[VAL_104:.*]] = vector.broadcast %[[VAL_6]] : i32 to vector<16xi32>
// CHECK:               %[[VAL_105:.*]] = arith.shli %[[VAL_103]], %[[VAL_104]] : vector<16xi32>
// CHECK:               %[[VAL_106:.*]] = vector.bitcast %[[VAL_105]] : vector<16xi32> to vector<16xf32>
// CHECK:               %[[VAL_107:.*]] = vector.load %[[VAL_22]]{{\[}}%[[VAL_10]], %[[VAL_1]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_108:.*]] = vector.bitcast %[[VAL_107]] : vector<16xbf16> to vector<16xi16>
// CHECK:               %[[VAL_109:.*]] = arith.extui %[[VAL_108]] : vector<16xi16> to vector<16xi32>
// CHECK:               %[[VAL_110:.*]] = vector.broadcast %[[VAL_6]] : i32 to vector<16xi32>
// CHECK:               %[[VAL_111:.*]] = arith.shli %[[VAL_109]], %[[VAL_110]] : vector<16xi32>
// CHECK:               %[[VAL_112:.*]] = vector.bitcast %[[VAL_111]] : vector<16xi32> to vector<16xf32>
// CHECK:               %[[VAL_113:.*]] = vector.load %[[VAL_22]]{{\[}}%[[VAL_4]], %[[VAL_1]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_114:.*]] = vector.bitcast %[[VAL_113]] : vector<16xbf16> to vector<16xi16>
// CHECK:               %[[VAL_115:.*]] = arith.extui %[[VAL_114]] : vector<16xi16> to vector<16xi32>
// CHECK:               %[[VAL_116:.*]] = vector.broadcast %[[VAL_6]] : i32 to vector<16xi32>
// CHECK:               %[[VAL_117:.*]] = arith.shli %[[VAL_115]], %[[VAL_116]] : vector<16xi32>
// CHECK:               %[[VAL_118:.*]] = vector.bitcast %[[VAL_117]] : vector<16xi32> to vector<16xf32>
// CHECK:               %[[VAL_119:.*]] = vector.load %[[VAL_22]]{{\[}}%[[VAL_3]], %[[VAL_1]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_120:.*]] = vector.bitcast %[[VAL_119]] : vector<16xbf16> to vector<16xi16>
// CHECK:               %[[VAL_121:.*]] = arith.extui %[[VAL_120]] : vector<16xi16> to vector<16xi32>
// CHECK:               %[[VAL_122:.*]] = vector.broadcast %[[VAL_6]] : i32 to vector<16xi32>
// CHECK:               %[[VAL_123:.*]] = arith.shli %[[VAL_121]], %[[VAL_122]] : vector<16xi32>
// CHECK:               %[[VAL_124:.*]] = vector.bitcast %[[VAL_123]] : vector<16xi32> to vector<16xf32>
// CHECK:               %[[VAL_125:.*]] = vector.load %[[VAL_22]]{{\[}}%[[VAL_2]], %[[VAL_1]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_126:.*]] = vector.bitcast %[[VAL_125]] : vector<16xbf16> to vector<16xi16>
// CHECK:               %[[VAL_127:.*]] = arith.extui %[[VAL_126]] : vector<16xi16> to vector<16xi32>
// CHECK:               %[[VAL_128:.*]] = vector.broadcast %[[VAL_6]] : i32 to vector<16xi32>
// CHECK:               %[[VAL_129:.*]] = arith.shli %[[VAL_127]], %[[VAL_128]] : vector<16xi32>
// CHECK:               %[[VAL_130:.*]] = vector.bitcast %[[VAL_129]] : vector<16xi32> to vector<16xf32>
// CHECK:               %[[VAL_131:.*]] = vector.load %[[VAL_22]]{{\[}}%[[VAL_15]], %[[VAL_0]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_132:.*]] = vector.bitcast %[[VAL_131]] : vector<16xbf16> to vector<16xi16>
// CHECK:               %[[VAL_133:.*]] = arith.extui %[[VAL_132]] : vector<16xi16> to vector<16xi32>
// CHECK:               %[[VAL_134:.*]] = vector.broadcast %[[VAL_6]] : i32 to vector<16xi32>
// CHECK:               %[[VAL_135:.*]] = arith.shli %[[VAL_133]], %[[VAL_134]] : vector<16xi32>
// CHECK:               %[[VAL_136:.*]] = vector.bitcast %[[VAL_135]] : vector<16xi32> to vector<16xf32>
// CHECK:               %[[VAL_137:.*]] = vector.load %[[VAL_22]]{{\[}}%[[VAL_9]], %[[VAL_0]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_138:.*]] = vector.bitcast %[[VAL_137]] : vector<16xbf16> to vector<16xi16>
// CHECK:               %[[VAL_139:.*]] = arith.extui %[[VAL_138]] : vector<16xi16> to vector<16xi32>
// CHECK:               %[[VAL_140:.*]] = vector.broadcast %[[VAL_6]] : i32 to vector<16xi32>
// CHECK:               %[[VAL_141:.*]] = arith.shli %[[VAL_139]], %[[VAL_140]] : vector<16xi32>
// CHECK:               %[[VAL_142:.*]] = vector.bitcast %[[VAL_141]] : vector<16xi32> to vector<16xf32>
// CHECK:               %[[VAL_143:.*]] = vector.load %[[VAL_22]]{{\[}}%[[VAL_10]], %[[VAL_0]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_144:.*]] = vector.bitcast %[[VAL_143]] : vector<16xbf16> to vector<16xi16>
// CHECK:               %[[VAL_145:.*]] = arith.extui %[[VAL_144]] : vector<16xi16> to vector<16xi32>
// CHECK:               %[[VAL_146:.*]] = vector.broadcast %[[VAL_6]] : i32 to vector<16xi32>
// CHECK:               %[[VAL_147:.*]] = arith.shli %[[VAL_145]], %[[VAL_146]] : vector<16xi32>
// CHECK:               %[[VAL_148:.*]] = vector.bitcast %[[VAL_147]] : vector<16xi32> to vector<16xf32>
// CHECK:               %[[VAL_149:.*]] = vector.load %[[VAL_22]]{{\[}}%[[VAL_4]], %[[VAL_0]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_150:.*]] = vector.bitcast %[[VAL_149]] : vector<16xbf16> to vector<16xi16>
// CHECK:               %[[VAL_151:.*]] = arith.extui %[[VAL_150]] : vector<16xi16> to vector<16xi32>
// CHECK:               %[[VAL_152:.*]] = vector.broadcast %[[VAL_6]] : i32 to vector<16xi32>
// CHECK:               %[[VAL_153:.*]] = arith.shli %[[VAL_151]], %[[VAL_152]] : vector<16xi32>
// CHECK:               %[[VAL_154:.*]] = vector.bitcast %[[VAL_153]] : vector<16xi32> to vector<16xf32>
// CHECK:               %[[VAL_155:.*]] = vector.load %[[VAL_22]]{{\[}}%[[VAL_3]], %[[VAL_0]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_156:.*]] = vector.bitcast %[[VAL_155]] : vector<16xbf16> to vector<16xi16>
// CHECK:               %[[VAL_157:.*]] = arith.extui %[[VAL_156]] : vector<16xi16> to vector<16xi32>
// CHECK:               %[[VAL_158:.*]] = vector.broadcast %[[VAL_6]] : i32 to vector<16xi32>
// CHECK:               %[[VAL_159:.*]] = arith.shli %[[VAL_157]], %[[VAL_158]] : vector<16xi32>
// CHECK:               %[[VAL_160:.*]] = vector.bitcast %[[VAL_159]] : vector<16xi32> to vector<16xf32>
// CHECK:               %[[VAL_161:.*]] = vector.load %[[VAL_22]]{{\[}}%[[VAL_2]], %[[VAL_0]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_162:.*]] = vector.bitcast %[[VAL_161]] : vector<16xbf16> to vector<16xi16>
// CHECK:               %[[VAL_163:.*]] = arith.extui %[[VAL_162]] : vector<16xi16> to vector<16xi32>
// CHECK:               %[[VAL_164:.*]] = vector.broadcast %[[VAL_6]] : i32 to vector<16xi32>
// CHECK:               %[[VAL_165:.*]] = arith.shli %[[VAL_163]], %[[VAL_164]] : vector<16xi32>
// CHECK:               %[[VAL_166:.*]] = vector.bitcast %[[VAL_165]] : vector<16xi32> to vector<16xf32>
// CHECK:               %[[VAL_167:.*]]:24 = scf.for %[[VAL_168:.*]] = %[[VAL_15]] to %[[VAL_10]] step %[[VAL_9]] iter_args(%[[VAL_169:.*]] = %[[VAL_28]], %[[VAL_170:.*]] = %[[VAL_34]], %[[VAL_171:.*]] = %[[VAL_40]], %[[VAL_172:.*]] = %[[VAL_46]], %[[VAL_173:.*]] = %[[VAL_52]], %[[VAL_174:.*]] = %[[VAL_58]], %[[VAL_175:.*]] = %[[VAL_64]], %[[VAL_176:.*]] = %[[VAL_70]], %[[VAL_177:.*]] = %[[VAL_76]], %[[VAL_178:.*]] = %[[VAL_82]], %[[VAL_179:.*]] = %[[VAL_88]], %[[VAL_180:.*]] = %[[VAL_94]], %[[VAL_181:.*]] = %[[VAL_100]], %[[VAL_182:.*]] = %[[VAL_106]], %[[VAL_183:.*]] = %[[VAL_112]], %[[VAL_184:.*]] = %[[VAL_118]], %[[VAL_185:.*]] = %[[VAL_124]], %[[VAL_186:.*]] = %[[VAL_130]], %[[VAL_187:.*]] = %[[VAL_136]], %[[VAL_188:.*]] = %[[VAL_142]], %[[VAL_189:.*]] = %[[VAL_148]], %[[VAL_190:.*]] = %[[VAL_154]], %[[VAL_191:.*]] = %[[VAL_160]], %[[VAL_192:.*]] = %[[VAL_166]]) -> (vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>) {
// CHECK:                 %[[VAL_193:.*]]:24 = scf.for %[[VAL_194:.*]] = %[[VAL_15]] to %[[VAL_8]] step %[[VAL_9]] iter_args(%[[VAL_195:.*]] = %[[VAL_169]], %[[VAL_196:.*]] = %[[VAL_170]], %[[VAL_197:.*]] = %[[VAL_171]], %[[VAL_198:.*]] = %[[VAL_172]], %[[VAL_199:.*]] = %[[VAL_173]], %[[VAL_200:.*]] = %[[VAL_174]], %[[VAL_201:.*]] = %[[VAL_175]], %[[VAL_202:.*]] = %[[VAL_176]], %[[VAL_203:.*]] = %[[VAL_177]], %[[VAL_204:.*]] = %[[VAL_178]], %[[VAL_205:.*]] = %[[VAL_179]], %[[VAL_206:.*]] = %[[VAL_180]], %[[VAL_207:.*]] = %[[VAL_181]], %[[VAL_208:.*]] = %[[VAL_182]], %[[VAL_209:.*]] = %[[VAL_183]], %[[VAL_210:.*]] = %[[VAL_184]], %[[VAL_211:.*]] = %[[VAL_185]], %[[VAL_212:.*]] = %[[VAL_186]], %[[VAL_213:.*]] = %[[VAL_187]], %[[VAL_214:.*]] = %[[VAL_188]], %[[VAL_215:.*]] = %[[VAL_189]], %[[VAL_216:.*]] = %[[VAL_190]], %[[VAL_217:.*]] = %[[VAL_191]], %[[VAL_218:.*]] = %[[VAL_192]]) -> (vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>) {
// CHECK:                   %[[VAL_219:.*]] = memref.subview %[[ARG0]]{{\[}}%[[VAL_168]], %[[VAL_20]], %[[VAL_194]], 0] [1, 6, 1, 2] [1, 1, 1, 1] : memref<2x24x16x2xbf16> to memref<1x6x1x2xbf16, strided<[768, 32, 2, 1], offset: ?>>
// CHECK:                   %[[VAL_220:.*]] = memref.subview %[[VAL_16]]{{\[}}%[[VAL_168]], %[[VAL_194]], %[[VAL_21]], 0] [1, 1, 64, 2] [1, 1, 1, 1] : memref<2x16x128x2xbf16> to memref<1x1x64x2xbf16, strided<[4096, 256, 2, 1], offset: ?>>
// CHECK:                   %[[VAL_221:.*]] = vector.load %[[VAL_220]]{{\[}}%[[VAL_15]], %[[VAL_15]], %[[VAL_15]], %[[VAL_15]]] : memref<1x1x64x2xbf16, strided<[4096, 256, 2, 1], offset: ?>>, vector<32xbf16>
// CHECK:                   %[[VAL_222:.*]] = vector.load %[[VAL_220]]{{\[}}%[[VAL_15]], %[[VAL_15]], %[[VAL_8]], %[[VAL_15]]] : memref<1x1x64x2xbf16, strided<[4096, 256, 2, 1], offset: ?>>, vector<32xbf16>
// CHECK:                   %[[VAL_223:.*]] = vector.load %[[VAL_220]]{{\[}}%[[VAL_15]], %[[VAL_15]], %[[VAL_1]], %[[VAL_15]]] : memref<1x1x64x2xbf16, strided<[4096, 256, 2, 1], offset: ?>>, vector<32xbf16>
// CHECK:                   %[[VAL_224:.*]] = vector.load %[[VAL_220]]{{\[}}%[[VAL_15]], %[[VAL_15]], %[[VAL_0]], %[[VAL_15]]] : memref<1x1x64x2xbf16, strided<[4096, 256, 2, 1], offset: ?>>, vector<32xbf16>
// CHECK:                   %[[VAL_225:.*]] = vector.load %[[VAL_219]]{{\[}}%[[VAL_15]], %[[VAL_15]], %[[VAL_15]], %[[VAL_15]]] : memref<1x6x1x2xbf16, strided<[768, 32, 2, 1], offset: ?>>, vector<2xbf16>
// CHECK:                   %[[VAL_226:.*]] = vector.bitcast %[[VAL_225]] : vector<2xbf16> to vector<1xi32>
// CHECK:                   %[[VAL_227:.*]] = vector.broadcast %[[VAL_226]] : vector<1xi32> to vector<16xi32>
// CHECK:                   %[[VAL_228:.*]] = vector.bitcast %[[VAL_227]] : vector<16xi32> to vector<32xbf16>
// CHECK:                   %[[VAL_229:.*]] = x86vector.avx512.dot %[[VAL_195]], %[[VAL_228]], %[[VAL_221]] : vector<32xbf16> -> vector<16xf32>
// CHECK:                   %[[VAL_230:.*]] = x86vector.avx512.dot %[[VAL_201]], %[[VAL_228]], %[[VAL_222]] : vector<32xbf16> -> vector<16xf32>
// CHECK:                   %[[VAL_231:.*]] = x86vector.avx512.dot %[[VAL_207]], %[[VAL_228]], %[[VAL_223]] : vector<32xbf16> -> vector<16xf32>
// CHECK:                   %[[VAL_232:.*]] = x86vector.avx512.dot %[[VAL_213]], %[[VAL_228]], %[[VAL_224]] : vector<32xbf16> -> vector<16xf32>
// CHECK:                   %[[VAL_233:.*]] = vector.load %[[VAL_219]]{{\[}}%[[VAL_15]], %[[VAL_9]], %[[VAL_15]], %[[VAL_15]]] : memref<1x6x1x2xbf16, strided<[768, 32, 2, 1], offset: ?>>, vector<2xbf16>
// CHECK:                   %[[VAL_234:.*]] = vector.bitcast %[[VAL_233]] : vector<2xbf16> to vector<1xi32>
// CHECK:                   %[[VAL_235:.*]] = vector.broadcast %[[VAL_234]] : vector<1xi32> to vector<16xi32>
// CHECK:                   %[[VAL_236:.*]] = vector.bitcast %[[VAL_235]] : vector<16xi32> to vector<32xbf16>
// CHECK:                   %[[VAL_237:.*]] = x86vector.avx512.dot %[[VAL_196]], %[[VAL_236]], %[[VAL_221]] : vector<32xbf16> -> vector<16xf32>
// CHECK:                   %[[VAL_238:.*]] = x86vector.avx512.dot %[[VAL_202]], %[[VAL_236]], %[[VAL_222]] : vector<32xbf16> -> vector<16xf32>
// CHECK:                   %[[VAL_239:.*]] = x86vector.avx512.dot %[[VAL_208]], %[[VAL_236]], %[[VAL_223]] : vector<32xbf16> -> vector<16xf32>
// CHECK:                   %[[VAL_240:.*]] = x86vector.avx512.dot %[[VAL_214]], %[[VAL_236]], %[[VAL_224]] : vector<32xbf16> -> vector<16xf32>
// CHECK:                   %[[VAL_241:.*]] = vector.load %[[VAL_219]]{{\[}}%[[VAL_15]], %[[VAL_10]], %[[VAL_15]], %[[VAL_15]]] : memref<1x6x1x2xbf16, strided<[768, 32, 2, 1], offset: ?>>, vector<2xbf16>
// CHECK:                   %[[VAL_242:.*]] = vector.bitcast %[[VAL_241]] : vector<2xbf16> to vector<1xi32>
// CHECK:                   %[[VAL_243:.*]] = vector.broadcast %[[VAL_242]] : vector<1xi32> to vector<16xi32>
// CHECK:                   %[[VAL_244:.*]] = vector.bitcast %[[VAL_243]] : vector<16xi32> to vector<32xbf16>
// CHECK:                   %[[VAL_245:.*]] = x86vector.avx512.dot %[[VAL_197]], %[[VAL_244]], %[[VAL_221]] : vector<32xbf16> -> vector<16xf32>
// CHECK:                   %[[VAL_246:.*]] = x86vector.avx512.dot %[[VAL_203]], %[[VAL_244]], %[[VAL_222]] : vector<32xbf16> -> vector<16xf32>
// CHECK:                   %[[VAL_247:.*]] = x86vector.avx512.dot %[[VAL_209]], %[[VAL_244]], %[[VAL_223]] : vector<32xbf16> -> vector<16xf32>
// CHECK:                   %[[VAL_248:.*]] = x86vector.avx512.dot %[[VAL_215]], %[[VAL_244]], %[[VAL_224]] : vector<32xbf16> -> vector<16xf32>
// CHECK:                   %[[VAL_249:.*]] = vector.load %[[VAL_219]]{{\[}}%[[VAL_15]], %[[VAL_4]], %[[VAL_15]], %[[VAL_15]]] : memref<1x6x1x2xbf16, strided<[768, 32, 2, 1], offset: ?>>, vector<2xbf16>
// CHECK:                   %[[VAL_250:.*]] = vector.bitcast %[[VAL_249]] : vector<2xbf16> to vector<1xi32>
// CHECK:                   %[[VAL_251:.*]] = vector.broadcast %[[VAL_250]] : vector<1xi32> to vector<16xi32>
// CHECK:                   %[[VAL_252:.*]] = vector.bitcast %[[VAL_251]] : vector<16xi32> to vector<32xbf16>
// CHECK:                   %[[VAL_253:.*]] = x86vector.avx512.dot %[[VAL_198]], %[[VAL_252]], %[[VAL_221]] : vector<32xbf16> -> vector<16xf32>
// CHECK:                   %[[VAL_254:.*]] = x86vector.avx512.dot %[[VAL_204]], %[[VAL_252]], %[[VAL_222]] : vector<32xbf16> -> vector<16xf32>
// CHECK:                   %[[VAL_255:.*]] = x86vector.avx512.dot %[[VAL_210]], %[[VAL_252]], %[[VAL_223]] : vector<32xbf16> -> vector<16xf32>
// CHECK:                   %[[VAL_256:.*]] = x86vector.avx512.dot %[[VAL_216]], %[[VAL_252]], %[[VAL_224]] : vector<32xbf16> -> vector<16xf32>
// CHECK:                   %[[VAL_257:.*]] = vector.load %[[VAL_219]]{{\[}}%[[VAL_15]], %[[VAL_3]], %[[VAL_15]], %[[VAL_15]]] : memref<1x6x1x2xbf16, strided<[768, 32, 2, 1], offset: ?>>, vector<2xbf16>
// CHECK:                   %[[VAL_258:.*]] = vector.bitcast %[[VAL_257]] : vector<2xbf16> to vector<1xi32>
// CHECK:                   %[[VAL_259:.*]] = vector.broadcast %[[VAL_258]] : vector<1xi32> to vector<16xi32>
// CHECK:                   %[[VAL_260:.*]] = vector.bitcast %[[VAL_259]] : vector<16xi32> to vector<32xbf16>
// CHECK:                   %[[VAL_261:.*]] = x86vector.avx512.dot %[[VAL_199]], %[[VAL_260]], %[[VAL_221]] : vector<32xbf16> -> vector<16xf32>
// CHECK:                   %[[VAL_262:.*]] = x86vector.avx512.dot %[[VAL_205]], %[[VAL_260]], %[[VAL_222]] : vector<32xbf16> -> vector<16xf32>
// CHECK:                   %[[VAL_263:.*]] = x86vector.avx512.dot %[[VAL_211]], %[[VAL_260]], %[[VAL_223]] : vector<32xbf16> -> vector<16xf32>
// CHECK:                   %[[VAL_264:.*]] = x86vector.avx512.dot %[[VAL_217]], %[[VAL_260]], %[[VAL_224]] : vector<32xbf16> -> vector<16xf32>
// CHECK:                   %[[VAL_265:.*]] = vector.load %[[VAL_219]]{{\[}}%[[VAL_15]], %[[VAL_2]], %[[VAL_15]], %[[VAL_15]]] : memref<1x6x1x2xbf16, strided<[768, 32, 2, 1], offset: ?>>, vector<2xbf16>
// CHECK:                   %[[VAL_266:.*]] = vector.bitcast %[[VAL_265]] : vector<2xbf16> to vector<1xi32>
// CHECK:                   %[[VAL_267:.*]] = vector.broadcast %[[VAL_266]] : vector<1xi32> to vector<16xi32>
// CHECK:                   %[[VAL_268:.*]] = vector.bitcast %[[VAL_267]] : vector<16xi32> to vector<32xbf16>
// CHECK:                   %[[VAL_269:.*]] = x86vector.avx512.dot %[[VAL_200]], %[[VAL_268]], %[[VAL_221]] : vector<32xbf16> -> vector<16xf32>
// CHECK:                   %[[VAL_270:.*]] = x86vector.avx512.dot %[[VAL_206]], %[[VAL_268]], %[[VAL_222]] : vector<32xbf16> -> vector<16xf32>
// CHECK:                   %[[VAL_271:.*]] = x86vector.avx512.dot %[[VAL_212]], %[[VAL_268]], %[[VAL_223]] : vector<32xbf16> -> vector<16xf32>
// CHECK:                   %[[VAL_272:.*]] = x86vector.avx512.dot %[[VAL_218]], %[[VAL_268]], %[[VAL_224]] : vector<32xbf16> -> vector<16xf32>
// CHECK:                   scf.yield %[[VAL_229]], %[[VAL_237]], %[[VAL_245]], %[[VAL_253]], %[[VAL_261]], %[[VAL_269]], %[[VAL_230]], %[[VAL_238]], %[[VAL_246]], %[[VAL_254]], %[[VAL_262]], %[[VAL_270]], %[[VAL_231]], %[[VAL_239]], %[[VAL_247]], %[[VAL_255]], %[[VAL_263]], %[[VAL_271]], %[[VAL_232]], %[[VAL_240]], %[[VAL_248]], %[[VAL_256]], %[[VAL_264]], %[[VAL_272]] : vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_273:.*]]#0, %[[VAL_273]]#1, %[[VAL_273]]#2, %[[VAL_273]]#3, %[[VAL_273]]#4, %[[VAL_273]]#5, %[[VAL_273]]#6, %[[VAL_273]]#7, %[[VAL_273]]#8, %[[VAL_273]]#9, %[[VAL_273]]#10, %[[VAL_273]]#11, %[[VAL_273]]#12, %[[VAL_273]]#13, %[[VAL_273]]#14, %[[VAL_273]]#15, %[[VAL_273]]#16, %[[VAL_273]]#17, %[[VAL_273]]#18, %[[VAL_273]]#19, %[[VAL_273]]#20, %[[VAL_273]]#21, %[[VAL_273]]#22, %[[VAL_273]]#23 : vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>
// CHECK:               }
// CHECK:               %[[VAL_274:.*]] = arith.truncf %[[VAL_275:.*]]#0 : vector<16xf32> to vector<16xbf16>
// CHECK:               vector.store %[[VAL_274]], %[[VAL_22]]{{\[}}%[[VAL_15]], %[[VAL_15]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_276:.*]] = arith.truncf %[[VAL_275]]#1 : vector<16xf32> to vector<16xbf16>
// CHECK:               vector.store %[[VAL_276]], %[[VAL_22]]{{\[}}%[[VAL_9]], %[[VAL_15]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_277:.*]] = arith.truncf %[[VAL_275]]#2 : vector<16xf32> to vector<16xbf16>
// CHECK:               vector.store %[[VAL_277]], %[[VAL_22]]{{\[}}%[[VAL_10]], %[[VAL_15]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_278:.*]] = arith.truncf %[[VAL_275]]#3 : vector<16xf32> to vector<16xbf16>
// CHECK:               vector.store %[[VAL_278]], %[[VAL_22]]{{\[}}%[[VAL_4]], %[[VAL_15]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_279:.*]] = arith.truncf %[[VAL_275]]#4 : vector<16xf32> to vector<16xbf16>
// CHECK:               vector.store %[[VAL_279]], %[[VAL_22]]{{\[}}%[[VAL_3]], %[[VAL_15]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_280:.*]] = arith.truncf %[[VAL_275]]#5 : vector<16xf32> to vector<16xbf16>
// CHECK:               vector.store %[[VAL_280]], %[[VAL_22]]{{\[}}%[[VAL_2]], %[[VAL_15]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_281:.*]] = arith.truncf %[[VAL_275]]#6 : vector<16xf32> to vector<16xbf16>
// CHECK:               vector.store %[[VAL_281]], %[[VAL_22]]{{\[}}%[[VAL_15]], %[[VAL_8]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_282:.*]] = arith.truncf %[[VAL_275]]#7 : vector<16xf32> to vector<16xbf16>
// CHECK:               vector.store %[[VAL_282]], %[[VAL_22]]{{\[}}%[[VAL_9]], %[[VAL_8]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_283:.*]] = arith.truncf %[[VAL_275]]#8 : vector<16xf32> to vector<16xbf16>
// CHECK:               vector.store %[[VAL_283]], %[[VAL_22]]{{\[}}%[[VAL_10]], %[[VAL_8]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_284:.*]] = arith.truncf %[[VAL_275]]#9 : vector<16xf32> to vector<16xbf16>
// CHECK:               vector.store %[[VAL_284]], %[[VAL_22]]{{\[}}%[[VAL_4]], %[[VAL_8]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_285:.*]] = arith.truncf %[[VAL_275]]#10 : vector<16xf32> to vector<16xbf16>
// CHECK:               vector.store %[[VAL_285]], %[[VAL_22]]{{\[}}%[[VAL_3]], %[[VAL_8]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_286:.*]] = arith.truncf %[[VAL_275]]#11 : vector<16xf32> to vector<16xbf16>
// CHECK:               vector.store %[[VAL_286]], %[[VAL_22]]{{\[}}%[[VAL_2]], %[[VAL_8]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_287:.*]] = arith.truncf %[[VAL_275]]#12 : vector<16xf32> to vector<16xbf16>
// CHECK:               vector.store %[[VAL_287]], %[[VAL_22]]{{\[}}%[[VAL_15]], %[[VAL_1]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_288:.*]] = arith.truncf %[[VAL_275]]#13 : vector<16xf32> to vector<16xbf16>
// CHECK:               vector.store %[[VAL_288]], %[[VAL_22]]{{\[}}%[[VAL_9]], %[[VAL_1]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_289:.*]] = arith.truncf %[[VAL_275]]#14 : vector<16xf32> to vector<16xbf16>
// CHECK:               vector.store %[[VAL_289]], %[[VAL_22]]{{\[}}%[[VAL_10]], %[[VAL_1]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_290:.*]] = arith.truncf %[[VAL_275]]#15 : vector<16xf32> to vector<16xbf16>
// CHECK:               vector.store %[[VAL_290]], %[[VAL_22]]{{\[}}%[[VAL_4]], %[[VAL_1]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_291:.*]] = arith.truncf %[[VAL_275]]#16 : vector<16xf32> to vector<16xbf16>
// CHECK:               vector.store %[[VAL_291]], %[[VAL_22]]{{\[}}%[[VAL_3]], %[[VAL_1]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_292:.*]] = arith.truncf %[[VAL_275]]#17 : vector<16xf32> to vector<16xbf16>
// CHECK:               vector.store %[[VAL_292]], %[[VAL_22]]{{\[}}%[[VAL_2]], %[[VAL_1]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_293:.*]] = arith.truncf %[[VAL_275]]#18 : vector<16xf32> to vector<16xbf16>
// CHECK:               vector.store %[[VAL_293]], %[[VAL_22]]{{\[}}%[[VAL_15]], %[[VAL_0]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_294:.*]] = arith.truncf %[[VAL_275]]#19 : vector<16xf32> to vector<16xbf16>
// CHECK:               vector.store %[[VAL_294]], %[[VAL_22]]{{\[}}%[[VAL_9]], %[[VAL_0]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_295:.*]] = arith.truncf %[[VAL_275]]#20 : vector<16xf32> to vector<16xbf16>
// CHECK:               vector.store %[[VAL_295]], %[[VAL_22]]{{\[}}%[[VAL_10]], %[[VAL_0]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_296:.*]] = arith.truncf %[[VAL_275]]#21 : vector<16xf32> to vector<16xbf16>
// CHECK:               vector.store %[[VAL_296]], %[[VAL_22]]{{\[}}%[[VAL_4]], %[[VAL_0]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_297:.*]] = arith.truncf %[[VAL_275]]#22 : vector<16xf32> to vector<16xbf16>
// CHECK:               vector.store %[[VAL_297]], %[[VAL_22]]{{\[}}%[[VAL_3]], %[[VAL_0]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:               %[[VAL_298:.*]] = arith.truncf %[[VAL_275]]#23 : vector<16xf32> to vector<16xbf16>
// CHECK:               vector.store %[[VAL_298]], %[[VAL_22]]{{\[}}%[[VAL_2]], %[[VAL_0]]] : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<16xbf16>
// CHECK:             }
// CHECK:           }
// CHECK:           return %[[VAL_17]] : memref<24x128xbf16>
// CHECK:         }

// -----
