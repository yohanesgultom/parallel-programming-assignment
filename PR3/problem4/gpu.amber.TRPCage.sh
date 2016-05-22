#!/bin/bash

echo ""
echo "TRPCAGE_PRODUCTION - 304 atoms GB"
echo "---------------------------------"
cd Amber_GPU_Benchmark_Suite/GB/TRPCage
export CUDA_VISIBLE_DEVICES=0
$AMBERHOME/bin/pmemd.cuda -O -o mdout.GPU -inf mdinfo.GPU -x mdcrd.GPU -r restrt.GPU -ref inpcrd
tail -100f mdinfo.GPU
