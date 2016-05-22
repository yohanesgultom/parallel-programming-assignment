#!/bin/bash

echo ""
echo "FACTOR_IX_PRODUCTION_NVE - 90,906 atoms PME"
echo "-------------------------------------------"
cd Amber_GPU_Benchmark_Suite/PME/FactorIX_production_NVE
export CUDA_VISIBLE_DEVICES=0
$AMBERHOME/bin/pmemd.cuda -O -o mdout.GPU -inf mdinfo.GPU -x mdcrd.GPU -r restrt.GPU
tail -100f mdinfo.GPU
