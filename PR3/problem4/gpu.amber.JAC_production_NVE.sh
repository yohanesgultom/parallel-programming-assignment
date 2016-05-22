#!/bin/bash

echo ""
echo "JAC_PRODUCTION_NVE - 23,558 atoms PME"
echo "-------------------------------------"
cd Amber_GPU_Benchmark_Suite/PME/JAC_production_NVE
export CUDA_VISIBLE_DEVICES=0
$AMBERHOME/bin/pmemd.cuda -O -o mdout.GPU -inf mdinfo.GPU -x mdcrd.GPU -r restrt.GPU
tail -100f mdinfo.GPU
