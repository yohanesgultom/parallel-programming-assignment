#!/bin/bash

echo ""
echo "MYOGLOBIN_PRODUCTION - 2,492 atoms GB"
echo "-------------------------------------"
cd Amber_GPU_Benchmark_Suite/GB/myoglobin
export CUDA_VISIBLE_DEVICES=0
$AMBERHOME/bin/pmemd.cuda -O -o mdout.GPU -inf mdinfo.GPU -x mdcrd.GPU -r restrt.GPU -ref inpcrd
tail -100f mdinfo.GPU
