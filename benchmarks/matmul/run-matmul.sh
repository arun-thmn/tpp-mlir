#!/bin/bash

source ../common.sh

# Compile driver. 
clang -O3 -emit-llvm -S matmul_driver.c
llc matmul_driver.ll

# Fire tpp compiler.
standalone-opt matmul_kernel.mlir -tpp-compiler="enable-xsmm-conversion" | mlir-translate -mlir-to-llvmir -o matmul_kernel.ll
llc matmul_kernel.ll

# Merge them.
unamestr=$(uname)
if [[ "$unamestr" == 'Darwin' ]]; then
  export DYLD_LIBRARY_PATH=$LIB_PATH
else
  export LD_LIBRARY_PATH=$LIB_PATH
fi

clang -O3 matmul_driver.s matmul_kernel.s -L$LIB_PATH -lstandalone_c_runner_utils -o matmul

# Execute and check result.
./matmul > result.txt 2>&1

if cat result.txt | grep "Result is correct" &> /dev/null ; then
  printf "${GREEN} OK ${NC} \n"
else
  printf "${RED} Oh NO ${NC} \n";
fi

rm matmul
rm *.s
rm *.ll
rm result.txt