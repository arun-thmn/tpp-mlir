// RUN: mlir-gen --kernel=args --seed=0 --float-type=f32 --batch=128 --layers=2304,768 --tiles=64,48,64 2>&1 | FileCheck %s --check-prefix=FP32
// RUN: mlir-gen --kernel=args --seed=0 --float-type=bf16 --batch=128 --layers=2304,768 --tiles=64,48,64 2>&1 | FileCheck %s --check-prefix=BF16
// RUN: mlir-gen --kernel=args --seed=0 --float-type=f16 --batch=128 --layers=2304,768 --tiles=64,48,64 2>&1 | FileCheck %s --check-prefix=FP16

// RUN: mlir-gen --kernel=args --seed=0 --float-type=mx-bf16 --batch=128 --layers=2304,768 --tiles=64,48,64 2>&1 | FileCheck %s --check-prefix=MXBF16-GENERIC
// RUN: mlir-gen --kernel=args --seed=0 --float-type=mx-i8 --batch=128 --layers=2304,768 --tiles=64,48,64 2>&1 | FileCheck %s --check-prefix=MXI8-GENERIC
// RUN: mlir-gen --kernel=args --seed=0 --float-type=mx-f16 --batch=128 --layers=2304,768 --tiles=64,48,64 2>&1 | FileCheck %s --check-prefix=MXF16-GENERIC

// RUN: mlir-gen --kernel=args --seed=0 --float-type=mx-bf16 --batch=128 --layers=2304,768 --tiles=64,48,64 --output=contract 2>&1 | FileCheck %s --check-prefix=MXBF16-CONTRACT
// RUN: mlir-gen --kernel=args --seed=0 --float-type=mx-i8 --batch=128 --layers=2304,768 --tiles=64,48,64 --output=contract 2>&1 | FileCheck %s --check-prefix=MXI8-CONTRACT
// RUN: mlir-gen --kernel=args --seed=0 --float-type=mx-f16 --batch=128 --layers=2304,768 --tiles=64,48,64 --output=contract 2>&1 | FileCheck %s --check-prefix=MXF16-CONTRACT

// FP32: // RUN{{.*}}tpp-run %s -n {{\d*}}
// FP32: // RUN{{.*}}-e entry -entry-point-result=void
// FP32: // BENCH_TOTAL_FLOPS: 452984832
// FP32-DAG: #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// FP32-DAG: #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// FP32-DAG: #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// FP32:     func.func @entry(%arg0: tensor<2x36x64x64xf32>, %arg1: tensor<16x36x64x48xf32>, %arg2: tensor<2x16x64x48xf32>) -> tensor<2x16x64x48xf32>
// FP32-NOT: alloc
// FP32:     linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
// FP32:         arith.mulf
// FP32:         arith.addf
// FP32-NOT: dealloc

// BF16: // RUN{{.*}}tpp-run %s -n {{\d*}}
// BF16: // RUN{{.*}}-e entry -entry-point-result=void
// BF16: // BENCH_TOTAL_FLOPS: 452984832
// BF16-DAG: #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// BF16-DAG: #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// BF16-DAG: #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// BF16:     func.func @entry(%arg0: tensor<2x36x64x64xbf16>, %arg1: tensor<16x36x64x48xbf16>, %arg2: tensor<2x16x64x48xbf16>) -> tensor<2x16x64x48xbf16>
// BF16-NOT: alloc
// BF16:     linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
// BF16:         arith.mulf
// BF16:         arith.addf
// BF16-NOT: dealloc

// FP16: // RUN{{.*}}tpp-run %s -n {{\d*}}
// FP16: // RUN{{.*}}-e entry -entry-point-result=void
// FP16: // BENCH_TOTAL_FLOPS: 452984832
// FP16-DAG: #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// FP16-DAG: #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// FP16-DAG: #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// FP16:     func.func @entry(%arg0: tensor<2x36x64x64xf16>, %arg1: tensor<16x36x64x48xf16>, %arg2: tensor<2x16x64x48xf16>) -> tensor<2x16x64x48xf16>
// FP16-NOT: alloc
// FP16:     linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
// FP16:         arith.mulf
// FP16:         arith.addf
// FP16-NOT: dealloc

// MXBF16-GENERIC: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// MXBF16-GENERIC: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// MXBF16-GENERIC: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// MXBF16-GENERIC-LABEL:   func.func @entry(
// MXBF16-GENERIC-SAME:                     %[[ARG0:.*]]: tensor<2x36x64x64xbf16>,
// MXBF16-GENERIC-SAME:                     %[[ARG1:.*]]: tensor<16x36x64x48xbf16>,
// MXBF16-GENERIC-SAME:                     %[[ARG2:.*]]: tensor<2x16x64x48xf32>) -> tensor<2x16x64x48xf32> {
// MXBF16-GENERIC:           %[[VAL_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%[[ARG0]], %[[ARG1]] : tensor<2x36x64x64xbf16>, tensor<16x36x64x48xbf16>) outs(%[[ARG2]] : tensor<2x16x64x48xf32>) {
// MXBF16-GENERIC:           ^bb0(%[[VAL_1:.*]]: bf16, %[[VAL_2:.*]]: bf16, %[[VAL_3:.*]]: f32):
// MXBF16-GENERIC:             %[[VAL_4:.*]] = arith.extf %[[VAL_1]] : bf16 to f32
// MXBF16-GENERIC:             %[[VAL_5:.*]] = arith.extf %[[VAL_2]] : bf16 to f32
// MXBF16-GENERIC:             %[[VAL_6:.*]] = arith.mulf %[[VAL_4]], %[[VAL_5]] : f32
// MXBF16-GENERIC:             %[[VAL_7:.*]] = arith.addf %[[VAL_3]], %[[VAL_6]] : f32
// MXBF16-GENERIC:             linalg.yield %[[VAL_7]] : f32
// MXBF16-GENERIC:           } -> tensor<2x16x64x48xf32>
// MXBF16-GENERIC:           return %[[VAL_0]] : tensor<2x16x64x48xf32>
// MXBF16-GENERIC:         }

// MXI8-GENERIC: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// MXI8-GENERIC: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// MXI8-GENERIC: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// MXI8-GENERIC-LABEL:   func.func @entry(
// MXI8-GENERIC-SAME:                     %[[ARG0:.*]]: tensor<2x36x64x64xi8>,
// MXI8-GENERIC-SAME:                     %[[ARG1:.*]]: tensor<16x36x64x48xi8>,
// MXI8-GENERIC-SAME:                     %[[ARG2:.*]]: tensor<2x16x64x48xi32>) -> tensor<2x16x64x48xi32> {
// MXI8-GENERIC:           %[[VAL_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%[[ARG0]], %[[ARG1]] : tensor<2x36x64x64xi8>, tensor<16x36x64x48xi8>) outs(%[[ARG2]] : tensor<2x16x64x48xi32>) {
// MXI8-GENERIC:           ^bb0(%[[VAL_1:.*]]: i8, %[[VAL_2:.*]]: i8, %[[VAL_3:.*]]: i32):
// MXI8-GENERIC:             %[[VAL_4:.*]] = arith.extsi %[[VAL_1]] : i8 to i32
// MXI8-GENERIC:             %[[VAL_5:.*]] = arith.extsi %[[VAL_2]] : i8 to i32
// MXI8-GENERIC:             %[[VAL_6:.*]] = arith.muli %[[VAL_4]], %[[VAL_5]] : i32
// MXI8-GENERIC:             %[[VAL_7:.*]] = arith.addi %[[VAL_3]], %[[VAL_6]] : i32
// MXI8-GENERIC:             linalg.yield %[[VAL_7]] : i32
// MXI8-GENERIC:           } -> tensor<2x16x64x48xi32>
// MXI8-GENERIC:           return %[[VAL_0]] : tensor<2x16x64x48xi32>
// MXI8-GENERIC:         }

// MXBF16-CONTRACT: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// MXBF16-CONTRACT: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// MXBF16-CONTRACT: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// MXBF16-CONTRACT-LABEL:   func.func @entry(
// MXBF16-CONTRACT-SAME:                     %[[ARG0:.*]]: tensor<2x36x64x64xbf16>,
// MXBF16-CONTRACT-SAME:                     %[[ARG1:.*]]: tensor<16x36x64x48xbf16>,
// MXBF16-CONTRACT-SAME:                     %[[ARG2:.*]]: tensor<2x16x64x48xf32>) -> tensor<2x16x64x48xf32> {
// MXBF16-CONTRACT:           %[[VAL_0:.*]] = linalg.contract indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]] ins(%[[ARG0]], %[[ARG1]] : tensor<2x36x64x64xbf16>, tensor<16x36x64x48xbf16>) outs(%[[ARG2]] : tensor<2x16x64x48xf32>) -> tensor<2x16x64x48xf32>
// MXBF16-CONTRACT:           return %[[VAL_0]] : tensor<2x16x64x48xf32>
// MXBF16-CONTRACT:         }

// MXI8-CONTRACT: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// MXI8-CONTRACT: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// MXI8-CONTRACT: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// MXI8-CONTRACT-LABEL:   func.func @entry(
// MXI8-CONTRACT-SAME:                     %[[ARG0:.*]]: tensor<2x36x64x64xi8>,
// MXI8-CONTRACT-SAME:                     %[[ARG1:.*]]: tensor<16x36x64x48xi8>,
// MXI8-CONTRACT-SAME:                     %[[ARG2:.*]]: tensor<2x16x64x48xi32>) -> tensor<2x16x64x48xi32> {
// MXI8-CONTRACT:           %[[VAL_0:.*]] = linalg.contract indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]] ins(%[[ARG0]], %[[ARG1]] : tensor<2x36x64x64xi8>, tensor<16x36x64x48xi8>) outs(%[[ARG2]] : tensor<2x16x64x48xi32>) -> tensor<2x16x64x48xi32>
// MXI8-CONTRACT:           return %[[VAL_0]] : tensor<2x16x64x48xi32>
// MXI8-CONTRACT:         }

// MXF16-GENERIC: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// MXF16-GENERIC: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// MXF16-GENERIC: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// MXF16-GENERIC-LABEL:   func.func @entry(
// MXF16-GENERIC-SAME:                     %[[ARG0:.*]]: tensor<2x36x64x64xf16>,
// MXF16-GENERIC-SAME:                     %[[ARG1:.*]]: tensor<16x36x64x48xf16>,
// MXF16-GENERIC-SAME:                     %[[ARG2:.*]]: tensor<2x16x64x48xf32>) -> tensor<2x16x64x48xf32> {
// MXF16-GENERIC:           %[[VAL_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%[[ARG0]], %[[ARG1]] : tensor<2x36x64x64xf16>, tensor<16x36x64x48xf16>) outs(%[[ARG2]] : tensor<2x16x64x48xf32>) {
// MXF16-GENERIC:           ^bb0(%[[VAL_1:.*]]: f16, %[[VAL_2:.*]]: f16, %[[VAL_3:.*]]: f32):
// MXF16-GENERIC:             %[[VAL_4:.*]] = arith.extf %[[VAL_1]] : f16 to f32
// MXF16-GENERIC:             %[[VAL_5:.*]] = arith.extf %[[VAL_2]] : f16 to f32
// MXF16-GENERIC:             %[[VAL_6:.*]] = arith.mulf %[[VAL_4]], %[[VAL_5]] : f32
// MXF16-GENERIC:             %[[VAL_7:.*]] = arith.addf %[[VAL_3]], %[[VAL_6]] : f32
// MXF16-GENERIC:             linalg.yield %[[VAL_7]] : f32
// MXF16-GENERIC:           } -> tensor<2x16x64x48xf32>
// MXF16-GENERIC:           return %[[VAL_0]] : tensor<2x16x64x48xf32>
// MXF16-GENERIC:         }

// MXF16-CONTRACT: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// MXF16-CONTRACT: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// MXF16-CONTRACT: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// MXF16-CONTRACT-LABEL:   func.func @entry(
// MXF16-CONTRACT-SAME:                     %[[ARG0:.*]]: tensor<2x36x64x64xf16>,
// MXF16-CONTRACT-SAME:                     %[[ARG1:.*]]: tensor<16x36x64x48xf16>,
// MXF16-CONTRACT-SAME:                     %[[ARG2:.*]]: tensor<2x16x64x48xf32>) -> tensor<2x16x64x48xf32> {
// MXF16-CONTRACT:           %[[VAL_0:.*]] = linalg.contract indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]] ins(%[[ARG0]], %[[ARG1]] : tensor<2x36x64x64xf16>, tensor<16x36x64x48xf16>) outs(%[[ARG2]] : tensor<2x16x64x48xf32>) -> tensor<2x16x64x48xf32>
// MXF16-CONTRACT:           return %[[VAL_0]] : tensor<2x16x64x48xf32>
// MXF16-CONTRACT:         }
